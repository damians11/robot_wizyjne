import socket

server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server.bind(('localhost', 12345))
server.listen(1)
conn, addr = server.accept()

while True:
    data = conn.recv(1024).decode()
    if not data:
        break
    print("Received:", data)
    conn.send("ACK".encode())
