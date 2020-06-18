# Original author: Tawn Kramer

import asyncore
import json
import re
import socket


def replace_float_notation(string):
    """
    Replace unity float notation for languages like
    French or German that use comma instead of dot.
    This convert the json sent by Unity to a valid one.
    Ex: "test": 1,2, "key": 2 -> "test": 1.2, "key": 2

    :param string: (str) The incorrect json string
    :return: (str) Valid JSON string
    """
    regex_french_notation = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+),'
    regex_end = r'"[a-zA-Z_]+":(?P<num>[0-9,E-]+)}'

    for regex in [regex_french_notation, regex_end]:
        matches = re.finditer(regex, string, re.MULTILINE)

        for match in matches:
            num = match.group('num').replace(',', '.')
            string = string.replace(match.group('num'), num)
    return string


class IMesgHandler(object):
    """
    Abstract class that represent a socket message handler.
    """
    def on_connect(self, socket_handler):
        pass

    def on_recv_message(self, message):
        pass

    def on_close(self):
        pass

    def on_disconnect(self):
        pass


class SimServer(asyncore.dispatcher):
    """
    Receives network connections and establishes handlers for each client.
    Each client connection is handled by a new instance of the SimHandler class.

    :param address: (str, int) (address, port)
    :param msg_handler: (socket message handler object)
    """
    def __init__(self, address, msg_handler):
        asyncore.dispatcher.__init__(self)

        # create a TCP socket to listen for connections
        self.create_socket(socket.AF_INET, socket.SOCK_STREAM)

        # in case we have shutdown recently, allow the os to reuse this address. helps when restarting
        self.set_reuse_addr()

        # let TCP stack know that we'd like to sit on this address and listen for connections
        self.bind(address)

        # confirm for users what address we are listening on
        self.address = self.socket.getsockname()
        print('Binding to', self.address)

        # let tcp stack know we plan to process one outstanding request to connect request each loop
        self.listen(5)

        # keep a pointer to our IMesgHandler handler
        self.msg_handler = msg_handler
        self.sim_handler = None

    def handle_accept(self):
        # Called when a client connects to our socket
        client_info = self.accept()

        print('Got a new client', client_info[1])

        # make a new steering handler to communicate with the client
        self.sim_handler = SimHandler(sock=client_info[0], msg_handler=self.msg_handler)

    def handle_close(self):
        print("Server shutdown")
        # Called then server is shutdown
        self.close()

        if self.msg_handler is not None:
            self.msg_handler.on_close()

        if self.sim_handler is not None:
            self.sim_handler.handle_close()


class SimHandler(asyncore.dispatcher):
    """
    Handles messages from a single TCP client.

    :param sock: (socket object)
    :param chunk_size: (int)
    :param msg_handler: (socket message handler object)
    """

    def __init__(self, sock, chunk_size=(16 * 1024), msg_handler=None):
        # we call our base class init
        asyncore.dispatcher.__init__(self, sock=sock)

        # msg_handler handles incoming messages
        self.msg_handler = msg_handler

        if msg_handler:
            msg_handler.on_connect(self)

        # chunk size is the max number of bytes to read per network packet
        self.chunk_size = chunk_size

        # we make an empty list of packets to send to the client here
        self.data_to_write = []

        # and image bytes is an empty list of partial bytes of the image as it comes in
        self.data_to_read = []

    def writable(self):
        """
          We want to write if we have received data.
        """
        response = bool(self.data_to_write)

        return response

    def queue_message(self, msg):
        json_msg = json.dumps(msg)
        self.data_to_write.append(json_msg)

    def handle_write(self):
        """
          Write as much as possible of the most recent message we have received.
          This is only called by async manager when the socket is in a writable state
          and when self.writable return true, that yes, we have data to send.
        """

        # pop the first element from the list. encode will make it into a byte stream
        data = self.data_to_write.pop(0).encode()

        # send a slice of that data, up to a max of the chunk_size
        sent = self.send(data[:self.chunk_size])

        # if we didn't send all the data..
        if sent < len(data):
            # then slick off the portion that remains to be sent
            remaining = data[sent:]

            # since we've popped it off the list, add it back to the list to send next
            # probably should change this to a deque...
            self.data_to_write.insert(0, remaining)

    def handle_read(self):
        """
          Read an incoming message from the client and put it into our outgoing queue.
          handle_read should only be called when the given socket has data ready to be
          processed.
        """

        # receive a chunK of data with the max size chunk_size from our client.
        data = self.recv(self.chunk_size)

        if len(data) == 0:
            # this only happens when the connection is dropped
            self.handle_close()
            return

        self.data_to_read.append(data.decode("utf-8"))

        messages = ''.join(self.data_to_read).split('\n')

        self.data_to_read = []

        for mesg in messages:
            if len(mesg) < 2:
                continue
            if mesg[0] == '{' and mesg[-1] == '}':
                self.handle_json_message(mesg)
            else:
                self.data_to_read.append(mesg)

    def handle_json_message(self, chunk):
        """
        We are expecing a json object.
        
        :param chunk: (str)
        """
        try:
            # convert data into a string with decode, and then load it as a json object
            chunk = replace_float_notation(chunk)
            json_obj = json.loads(chunk)
        except Exception as e:
            # something bad happened, usually malformed json packet. jump back to idle and hope things continue
            print(e, 'failed to read json ', chunk)
            return

        try:
            if self.msg_handler:
                self.msg_handler.on_recv_message(json_obj)
        except Exception as e:
            print(e, '>>> failure during on_recv_message:', chunk)

    def handle_close(self):
        # when client drops or closes connection
        if self.msg_handler is not None:
            self.msg_handler.on_disconnect()
            self.msg_handler = None
            print('Connection dropped')

        self.close()
