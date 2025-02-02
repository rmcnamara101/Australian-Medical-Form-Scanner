import http.server
import socketserver

class CORSRequestHandler(http.server.SimpleHTTPRequestHandler):
    def end_headers(self):
        # Allow all origins (for development purposes)
        self.send_header("Access-Control-Allow-Origin", "*")
        http.server.SimpleHTTPRequestHandler.end_headers(self)

PORT = 8000
Handler = CORSRequestHandler

with socketserver.TCPServer(("", PORT), Handler) as httpd:
    print("Serving at port", PORT)
    httpd.serve_forever()
