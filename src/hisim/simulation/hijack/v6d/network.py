"""Limit hostname address selection to native v6d peer advertisements."""

import ipaddress
import os

from hisim.hook import BaseHook
from hisim.utils import get_logger

logger = get_logger()


class _PeerSocket:
    def __init__(self, original):
        self._original = original

    def __getattr__(self, name):
        return getattr(self._original, name)

    def gethostbyname(self, hostname):
        if hostname == self._original.gethostname():
            value = os.environ.get("ALIYUN_ECI_ETH1_IP", "").strip()
            if value:
                try:
                    address = ipaddress.IPv4Address(value)
                except ipaddress.AddressValueError as error:
                    raise RuntimeError(
                        "ALIYUN_ECI_ETH1_IP must be a usable IPv4"
                    ) from error
                if (
                    address.is_loopback
                    or address.is_unspecified
                    or address.is_multicast
                    or address.is_link_local
                ):
                    raise RuntimeError("ALIYUN_ECI_ETH1_IP must be a usable IPv4")
                logger.info(
                    "[v6d-network] advertise %s via ALIYUN_ECI_ETH1_IP", address
                )
                return str(address)
        return self._original.gethostbyname(hostname)


class M_V6dPeerAddressHook(BaseHook):
    HOOK_MODULE_NAME = r"^v6d\.server\.peers\.(vineyard|tiered_vineyard)\.peer$"
    REGEX = True

    @classmethod
    def hook(cls, target):
        # Replace the module's socket reference, never socket.gethostbyname itself.
        # getaddrinfo / KVCM subscriber reporter identity remain native.
        if os.environ.get("ALIYUN_ECI_ETH1_IP", "").strip() and not isinstance(
            target.socket, _PeerSocket
        ):
            target.socket = _PeerSocket(target.socket)
