import urllib.request

url = (
    "https://abcd-1234.ngrok.io/webhook/whatsapp"
    "?hub.mode=subscribe"
    "&hub.verify_token=123456789"
    "&hub.challenge=999999"
)

with urllib.request.urlopen(url) as response:
    print("Status:", response.status)
    print("Body:", response.read().decode())
