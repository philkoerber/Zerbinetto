#!/usr/bin/env python3
"""
GitHub Webhook Server for Zerbinetto Chess Bot
This server listens for push events to the main branch and triggers deployment
"""

import http.server
import socketserver
import json
import subprocess
import hmac
import hashlib
import os
import logging
from urllib.parse import urlparse, parse_qs

# Configuration
WEBHOOK_SECRET = "zerbinetto_webhook_secret"  # You'll set this in GitHub
WEBHOOK_PORT = 9001  # Different port from your website webhook
DEPLOY_SCRIPT = "/opt/zerbinetto/scripts/deploy.sh"
PROJECT_DIR = "/opt/zerbinetto"

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('/opt/zerbinetto/webhook.log'),
        logging.StreamHandler()
    ]
)

class ZerbinettoWebhookHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        """Handle POST requests from GitHub webhooks"""
        try:
            # Get the content length
            content_length = int(self.headers.get('Content-Length', 0))
            if content_length == 0:
                self.send_error(400, "No content")
                return

            # Read the request body
            body = self.rfile.read(content_length)
            
            # Verify the webhook signature (security)
            signature = self.headers.get('X-Hub-Signature-256', '')
            if not self.verify_signature(body, signature):
                logging.warning("Invalid webhook signature")
                self.send_error(401, "Invalid signature")
                return

            # Parse the JSON payload
            try:
                payload = json.loads(body.decode('utf-8'))
            except json.JSONDecodeError:
                logging.error("Invalid JSON payload")
                self.send_error(400, "Invalid JSON")
                return

            # Check if this is a push event to the main branch
            if (payload.get('ref') == 'refs/heads/main' and 
                payload.get('repository', {}).get('name') == 'Zerbinetto'):
                
                logging.info("Push to main branch detected, triggering Zerbinetto deployment")
                
                # Trigger deployment
                self.trigger_deployment()
                
                # Send success response
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"status": "success", "message": "Zerbinetto deployment triggered"}
                self.wfile.write(json.dumps(response).encode())
                
            else:
                logging.info("Webhook received but not a push to Zerbinetto main branch")
                self.send_response(200)
                self.send_header('Content-type', 'application/json')
                self.end_headers()
                response = {"status": "ignored", "message": "Not a push to Zerbinetto main branch"}
                self.wfile.write(json.dumps(response).encode())
                
        except Exception as e:
            logging.error(f"Error processing webhook: {str(e)}")
            self.send_error(500, "Internal server error")

    def verify_signature(self, body, signature):
        """Verify the GitHub webhook signature"""
        if not signature or not WEBHOOK_SECRET:
            return False
        
        expected_signature = 'sha256=' + hmac.new(
            WEBHOOK_SECRET.encode('utf-8'),
            body,
            hashlib.sha256
        ).hexdigest()
        
        return hmac.compare_digest(signature, expected_signature)

    def trigger_deployment(self):
        """Execute the deployment script"""
        try:
            logging.info("Executing Zerbinetto deployment script")
            
            # Change to project directory
            os.chdir(PROJECT_DIR)
            
            result = subprocess.run(
                [DEPLOY_SCRIPT, 'update'],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            if result.returncode == 0:
                logging.info("Zerbinetto deployment completed successfully")
                logging.info(f"Deployment output: {result.stdout}")
            else:
                logging.error(f"Zerbinetto deployment failed with return code {result.returncode}")
                logging.error(f"Deployment error: {result.stderr}")
                
        except subprocess.TimeoutExpired:
            logging.error("Zerbinetto deployment script timed out")
        except Exception as e:
            logging.error(f"Error executing Zerbinetto deployment script: {str(e)}")

    def do_GET(self):
        """Handle GET requests (health check)"""
        self.send_response(200)
        self.send_header('Content-type', 'text/plain')
        self.end_headers()
        self.wfile.write(b"Zerbinetto GitHub Webhook Server is running")

    def log_message(self, format, *args):
        """Override to use our logging configuration"""
        logging.info(f"{self.address_string()} - {format % args}")

def main():
    """Start the webhook server"""
    try:
        with socketserver.TCPServer(("", WEBHOOK_PORT), ZerbinettoWebhookHandler) as httpd:
            logging.info(f"Starting Zerbinetto webhook server on port {WEBHOOK_PORT}")
            logging.info("Press Ctrl+C to stop the server")
            httpd.serve_forever()
    except KeyboardInterrupt:
        logging.info("Shutting down Zerbinetto webhook server")
    except Exception as e:
        logging.error(f"Error starting server: {str(e)}")

if __name__ == "__main__":
    main()
