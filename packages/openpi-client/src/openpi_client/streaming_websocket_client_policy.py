from __future__ import annotations

import logging
import time
import threading
import queue
from typing import Dict, Optional, Tuple, Union, Any
import copy
from typing_extensions import override
import websockets.sync.client
import websockets.exceptions
import numpy as np
from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy

class WebsocketClientPolicy(_base_policy.BasePolicy):
    
    _STREAM_END_SENTINEL = object()
    
    def __init__(self, host: str = "0.0.0.0", port: Optional[int] = None, api_key: Optional[str] = None) -> None:
        self._uri = f"ws://{host}"
        if port is not None:
            self._uri += f":{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._action_queue: queue.Queue = queue.Queue(maxsize=100)
        self._receiver_thread: Optional[threading.Thread] = None
        self._stream_active: bool = True 
        self._lock = threading.Lock()
        self._request_counter = 0
        self._request_send_times: Dict[int, float] = {}
        self._request_send_stats: Dict[int, Dict[str, float]] = {}
        self._ws: Optional[websockets.sync.client.ClientConnection] = None
        _, self._server_metadata = self._wait_for_metadata_only()
        self._receiver_thread = threading.Thread(target=self._stream_receiver_loop, daemon=True)
        self._receiver_thread.start()       
        time.sleep(0.1) 


    def get_server_metadata(self) -> Dict:
        return self._server_metadata
    


    def _wait_for_metadata_only(self) -> Tuple[None, Dict]:
        logging.info(f"Waiting for server at {self._uri} for metadata...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None 
                with websockets.sync.client.connect(
                    self._uri, compression=None, max_size=None, additional_headers=headers
                ) as conn:
                   
                    metadata = msgpack_numpy.unpackb(conn.recv())
                    logging.info("Metadata successfully received.")
                    return None, metadata
                
            except ConnectionRefusedError:
                
                logging.info("Still waiting for server...")
                time.sleep(5)
            except websockets.exceptions.ConnectionClosedOK:
                
                logging.info("Metadata connection closed gracefully, retrying to connect...")
                time.sleep(5)
            except websockets.exceptions.ConnectionClosedError as e:
                
                logging.warning(f"Metadata connection closed unexpectedly: {e}. Retrying...")
                time.sleep(5)
            except Exception as e:
                
                logging.warning(f"Connection attempt failed with error: {e}. Retrying...")
                time.sleep(5)

    def _ensure_connection(self) -> bool:
        if self._ws is not None:
            return True 

        logging.info(f"Establishing new connection to {self._uri}...")
        try:
            headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
            self._ws = websockets.sync.client.connect(
                self._uri, compression=None, max_size=None, additional_headers=headers
            )
            
            try:
                _ = self._ws.recv(timeout=1.0) 
            except Exception:
                pass 
                
            logging.info("Connection established and initial message/metadata received.")
            return True
        except Exception as e:
            logging.error(f"Failed to establish connection: {e}", exc_info=True)
            self._ws = None 
            return False

  
    def get_left_queue_actions(self) -> np.ndarray:
      
        with self._action_queue.mutex:
            if not self._action_queue.queue:
                print("Queue is empty, returning inf.")
                return np.zeros(7, dtype=np.float32)
            queue_snapshot = list(self._action_queue.queue)
        summed_action = None
        
        for element_dict in queue_snapshot:
            if isinstance(element_dict, dict) and "actions" in element_dict:
                try:
                    action_np = np.asarray(element_dict["actions"], dtype=np.float32).flatten()
                except (ValueError, TypeError):
                    continue

                if summed_action is None:
                    summed_action = np.zeros_like(action_np)

                curr_size = action_np.size
                acc_size = summed_action.size

                if curr_size > acc_size:
                    new_sum = np.zeros(curr_size, dtype=np.float32)
                    new_sum[:acc_size] = summed_action
                    summed_action = new_sum

                summed_action[:curr_size] += action_np

        if summed_action is None:
            return np.zeros(7, dtype=np.float32)

        return summed_action[:7].astype(np.float32)



    def _stream_receiver_loop(self) -> None:
        logging.info("[Receiver Thread] Stream thread started and running continuously.")
        
        while self._stream_active:
            if not self._ensure_connection():
                continue

            try:
                response = self._ws.recv()
                recv_time = time.monotonic()

                if isinstance(response, str):
                    logging.error("[Receiver Thread] Server returned an error frame:\n%s", response)
                    self._action_queue.put({"server_error": response})
                    with self._lock:
                        if self._ws:
                            try:
                                self._ws.close()
                            except Exception:
                                pass
                        self._ws = None
                    continue
                downlink_payload_bytes = len(response)
                unpack_start = time.monotonic()
                unpacked_response = msgpack_numpy.unpackb(response)
                client_unpack_ms = (time.monotonic() - unpack_start) * 1000

                if isinstance(unpacked_response, dict):
                    transport_timing = unpacked_response.setdefault("transport_timing", {})
                    transport_timing["client_downlink_payload_bytes"] = downlink_payload_bytes
                    transport_timing["client_unpack_ms"] = client_unpack_ms
                    request_id = unpacked_response.get("request_id", transport_timing.get("request_id"))
                    if request_id in self._request_send_times:
                        transport_timing["client_e2e_ms"] = (recv_time - self._request_send_times[request_id]) * 1000
                    if request_id in self._request_send_stats:
                        transport_timing.update(self._request_send_stats[request_id])

                if self._action_queue.full():
                    logging.info(f"Queue Is Full !")
                self._action_queue.put(unpacked_response)
                logging.debug(f"[Receiver Thread] Action received. Size: {self._action_queue.qsize()}")

            except websockets.exceptions.ConnectionClosed as e:
                logging.warning("[Receiver Thread] Connection closed. Attempting to re-establish: %s", e)
                self._action_queue.put({"server_error": "websocket connection closed: {}".format(e)})
                with self._lock:
                     if self._ws:
                          try:
                               self._ws.close()
                          except Exception:
                               pass
                     self._ws = None 
                time.sleep(1) 
            except Exception as e:
                if self._stream_active: 
                    logging.error(f"[Receiver Thread] Error during stream reception: {e}", exc_info=True)
                    self._action_queue.put({"server_error": str(e)})

                with self._lock:
                     self._ws = None 
              
        logging.info("[Receiver Thread] Exiting due to stream_active=False.")


    @override
    def infer(self, obs: Dict,new_task: bool) -> None:  
        # if it is a new task, we clear the queue to avoid executing old actions from the previous task.
        if new_task:
            with self._action_queue.mutex:
                self._action_queue.queue.clear() 

        logging.info(f"[Client] : queue empty: {self._action_queue.empty()}")
        
        with self._lock:
            if not self._ensure_connection():
                logging.warning("[client] Cannot send observation: Connection is down.")
                return 

            try:
                obs_with_metadata = copy.copy(obs)
                request_id = obs_with_metadata.get("__request_id__")
                if request_id is None:
                    request_id = obs_with_metadata.get("request_id")
                if request_id is None:
                    request_id = self._request_counter
                    self._request_counter += 1
                telemetry = dict(obs_with_metadata.get("__telemetry", {}))
                telemetry["request_id"] = request_id
                obs_with_metadata["__telemetry"] = telemetry

                pack_start = time.monotonic()
                data = self._packer.pack(obs_with_metadata)
                client_pack_ms = (time.monotonic() - pack_start) * 1000
                send_start = time.monotonic()
                self._ws.send(data)
                client_send_ms = (time.monotonic() - send_start) * 1000
                self._request_send_times[request_id] = send_start
                self._request_send_stats[request_id] = {
                    "client_pack_ms": client_pack_ms,
                    "client_send_ms": client_send_ms,
                    "client_uplink_payload_bytes": len(data),
                }
                logging.info(
                    "[ClientTransport] request_id=%s uplink_payload=%s bytes pack=%.3f ms send=%.3f ms",
                    request_id,
                    len(data),
                    client_pack_ms,
                    client_send_ms,
                )

            except websockets.exceptions.ConnectionClosed as e:
                logging.warning("Connection closed during send. _Receiver_loop will handle re-establishment: %s", e)
                self._action_queue.put({"server_error": "websocket connection closed during send: {}".format(e)})
                self._ws = None
            except Exception as e:
                logging.error(f"Unexpected error during send: {e}")
                self._action_queue.put({"server_error": "send failed: {}".format(e)})
                self._ws = None
                return
                
    def clear_action_queue(self) -> int:
        """Drop all queued actions without draining via get_next_action."""
        q = self._action_queue
        with q.mutex:
            dropped = len(q.queue)
            q.queue.clear()
            if hasattr(q, "unfinished_tasks"):
                q.unfinished_tasks = max(0, q.unfinished_tasks - dropped)
            if hasattr(q, "all_tasks_done"):
                q.all_tasks_done.notify_all()
        self._dropped_stale_actions = getattr(self, "_dropped_stale_actions", 0) + dropped
        return dropped

    def get_queue_length(self) -> int:
        with self._action_queue.mutex:
            return len(self._action_queue.queue)

    @staticmethod
    def _coerce_libero_action(item: Dict) -> Optional[Dict]:
        if "actions" not in item:
            return item
        try:
            action_np = np.asarray(item["actions"], dtype=np.float32).reshape(-1)
        except (TypeError, ValueError):
            logging.warning("Dropping action with non-numeric payload.")
            return None
        if action_np.size < 7:
            logging.warning("Dropping action with fewer than 7 dimensions: shape=%s", action_np.shape)
            return None
        action_np = action_np[:7]
        if not np.all(np.isfinite(action_np)):
            logging.warning("Dropping action containing NaN/Inf values.")
            return None
        result = dict(item)
        result["actions"] = action_np
        return result

    def get_next_action(self, timeout: Optional[float] = 5.0, request_id: Optional[str] = None) -> Union[Dict, None]:
        deadline = None if timeout is None else time.monotonic() + max(timeout, 0.0)
        while True:
            remaining = None if deadline is None else deadline - time.monotonic()
            if remaining is not None and remaining <= 0:
                return None
            try:
                item = self._action_queue.get(timeout=remaining)
            except queue.Empty:
                return None
            except Exception as e:
                logging.error("Error getting action from queue: %s", e, exc_info=True)
                return None

            try:
                if not isinstance(item, dict):
                    return item

                item_request_id = item.get("request_id")
                if item_request_id is None:
                    item_request_id = item.get("transport_timing", {}).get("request_id")
                if request_id is not None and item_request_id is not None and item_request_id != request_id:
                    self._dropped_stale_actions = getattr(self, "_dropped_stale_actions", 0) + 1
                    logging.info("Dropped stale action request_id=%s while waiting for %s", item_request_id, request_id)
                    continue

                if "norm_exceeded" in item or "server_error" in item:
                    return item

                coerced = self._coerce_libero_action(item)
                if coerced is None:
                    self._dropped_stale_actions = getattr(self, "_dropped_stale_actions", 0) + 1
                    continue
                return coerced
            finally:
                try:
                    self._action_queue.task_done()
                except ValueError:
                    pass
