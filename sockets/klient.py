import socket

client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client.connect(('localhost', 12345))

while True:
    msg = input("Send: ")
    client.send(msg.encode())
    data = client.recv(1024).decode()
    print("Response:", data)
