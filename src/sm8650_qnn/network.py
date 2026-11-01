"""Block Python network access during local stages. SPDX-License-Identifier: CC0-1.0."""
import socket


def disable_network():
    def denied(*args, **kwargs):
        raise PermissionError("Network disabled for local conversion; use the fetch stage to download assets")

    socket.create_connection = denied
    socket.socket.connect = denied
    socket.socket.connect_ex = denied
    socket.socket.sendto = denied
