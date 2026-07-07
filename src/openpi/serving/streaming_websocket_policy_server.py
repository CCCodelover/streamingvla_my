import asyncio
import http
import logging
import time
import traceback

from openpi_client import base_policy as _base_policy
from openpi_client import msgpack_numpy
from typing import Iterator, Optional
import websockets.asyncio.server as _server
import websockets.frames

logger = logging.getLogger(__name__)


class WebsocketPolicyServer:
    """Serves a policy using the websocket protocol. See websocket_client_policy.py for a client implementation.

    Currently only implements the `load` and `infer` methods.
    """

    def __init__(
        self,
        policy: _base_policy.BasePolicy,
        host: str = "0.0.0.0",
        port: Optional[int] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        self._policy = policy
        self._host = host
        self._port = port
        self._metadata = metadata or {}
        logging.getLogger("websockets.server").setLevel(logging.INFO)

    def serve_forever(self) -> None:
        asyncio.run(self.run())

    async def run(self):
        async with _server.serve(
            self._handler,
            self._host,
            self._port,
            compression=None,
            max_size=None,
            process_request=_health_check,
        ) as server:
            await server.serve_forever()
  
    async def _handler(self, websocket: _server.ServerConnection):
        logger.info(f"Connection from {websocket.remote_address} opened")
        packer = msgpack_numpy.Packer()

        await websocket.send(packer.pack(self._metadata))
        index=0
        while True:
            try:
                recv_payload = await websocket.recv()
                uplink_payload_bytes = len(recv_payload) if isinstance(recv_payload, bytes) else len(recv_payload.encode("utf-8"))
                unpack_start = time.monotonic()
                obs = msgpack_numpy.unpackb(recv_payload)
                server_unpack_ms = (time.monotonic() - unpack_start) * 1000
                request_metadata = obs.get("__telemetry", {}) if isinstance(obs, dict) else {}
                request_id = None
                execution_horizon = None
                if isinstance(obs, dict):
                    request_id = obs.pop("__request_id__", None)
                    if request_id is None:
                        request_id = obs.pop("request_id", None)
                    if request_id is None:
                        request_id = request_metadata.get("request_id")
                    raw_horizon = obs.pop("execution_horizon", None)
                    if raw_horizon is not None:
                        try:
                            execution_horizon = max(1, int(raw_horizon))
                        except (TypeError, ValueError):
                            execution_horizon = None

                action_stream: Iterator[dict] = self._policy.streaming_infer(obs)
                policy_start = time.monotonic()
                sent_actions = 0
                for action in action_stream:
                    if not isinstance(action, dict):
                        continue
                    model_timing = action.pop("model_timing", {})
                    if not isinstance(model_timing, dict):
                        model_timing = {}
                    action["request_id"] = request_id
                    if "actions" in action:
                        server_policy_to_action_ms = (time.monotonic() - policy_start) * 1000
                        server_timing = dict(model_timing)
                        server_timing.setdefault("infer_ms", server_policy_to_action_ms)
                        server_timing.setdefault("server_policy_time_ms", server_policy_to_action_ms)
                        server_timing.setdefault("server_unpack_time_ms", server_unpack_ms)
                        server_timing.setdefault("server_pack_time_ms", 0.0)
                        action["server_timing"] = server_timing
                        action["transport_timing"] = {
                            "request_id": request_id,
                            "uplink_payload_bytes": uplink_payload_bytes,
                            "server_unpack_ms": server_unpack_ms,
                            "server_unpack_time_ms": server_unpack_ms,
                            "server_policy_to_action_ms": server_policy_to_action_ms,
                            "server_policy_time_ms": server_policy_to_action_ms,
                        }
                        action["index"] = index
                        index=index+1

                        pack_start = time.monotonic()
                        action["transport_timing"]["server_pack_ms"] = 0.0
                        action["transport_timing"]["server_pack_time_ms"] = 0.0
                        packed_action = packer.pack(action)
                        server_pack_ms = (time.monotonic() - pack_start) * 1000
                        action["server_timing"]["server_pack_time_ms"] = server_pack_ms
                        action["transport_timing"]["server_pack_ms"] = server_pack_ms
                        action["transport_timing"]["server_pack_time_ms"] = server_pack_ms
                        action["transport_timing"]["downlink_payload_bytes"] = len(packed_action)
                        packed_action = packer.pack(action)
                        await websocket.send(packed_action)

                        policy_start = time.monotonic()
                        sent_actions += 1
                        if execution_horizon is not None and sent_actions >= execution_horizon:
                            break

                    if "norm_exceeded" in action:
                        print("[Server] Norm exceeded, send exceed signal")
                        server_timing = dict(model_timing)
                        server_timing.setdefault("server_unpack_time_ms", server_unpack_ms)
                        server_timing.setdefault("server_policy_time_ms", (time.monotonic() - policy_start) * 1000)
                        action["server_timing"] = server_timing
                        action["transport_timing"] = {
                            "request_id": request_id,
                            "uplink_payload_bytes": uplink_payload_bytes,
                            "server_unpack_ms": server_unpack_ms,
                            "server_unpack_time_ms": server_unpack_ms,
                        }
                        packed_action = packer.pack(action)
                        action["transport_timing"]["downlink_payload_bytes"] = len(packed_action)
                        packed_action = packer.pack(action)
                        await websocket.send(packed_action)
                
            except websockets.ConnectionClosed:
                logger.info(f"Connection from {websocket.remote_address} closed")
                break
            except Exception:
                await websocket.send(traceback.format_exc())
                await websocket.close(
                    code=websockets.frames.CloseCode.INTERNAL_ERROR,
                    reason="Internal server error. Traceback included in previous frame.",
                )
                raise
    

def _health_check(connection: _server.ServerConnection, request: _server.Request) -> _server.Response | None:
    if request.path == "/healthz":
        return connection.respond(http.HTTPStatus.OK, "OK\n")
    # Continue with the normal request handling.
    return None











