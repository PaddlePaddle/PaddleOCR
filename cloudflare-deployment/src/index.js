/**
 * Cloudflare Worker that routes requests to PaddleOCR Container
 * This is the entry point for the Worker
 */

export default {
  async fetch(request, env, ctx) {
    const url = new URL(request.url);

    // Health check endpoint
    if (url.pathname === '/health') {
      return new Response(JSON.stringify({
        status: 'healthy',
        worker: 'active',
        timestamp: new Date().toISOString()
      }), {
        headers: { 'Content-Type': 'application/json' }
      });
    }

    // Route to container for OCR endpoints
    if (url.pathname.startsWith('/ocr') || url.pathname === '/') {
      try {
        // Get or create Durable Object instance
        const id = env.PADDLE_OCR.idFromName('ocr-instance');
        const stub = env.PADDLE_OCR.get(id);

        // Forward request to container
        return await stub.fetch(request);
      } catch (error) {
        return new Response(JSON.stringify({
          success: false,
          error: 'Failed to connect to OCR service',
          detail: error.message
        }), {
          status: 500,
          headers: { 'Content-Type': 'application/json' }
        });
      }
    }

    // 404 for unknown routes
    return new Response('Not Found', { status: 404 });
  }
};

/**
 * Durable Object class for PaddleOCR Container
 * This manages the lifecycle of the container instance
 */
export class PaddleOCRContainer {
  constructor(state, env) {
    this.state = state;
    this.env = env;
  }

  async fetch(request) {
    // Forward all requests to the container's HTTP server
    // The container is running on port 8000 (defined in Dockerfile/app.py)
    return await fetch('http://localhost:8000' + new URL(request.url).pathname, {
      method: request.method,
      headers: request.headers,
      body: request.body,
    });
  }
}
