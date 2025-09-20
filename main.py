import io, os, json, asyncio
from typing import Optional, List, Union
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import httpx
from paddleocr import PaddleOCR
from PIL import Image

# Single OCR instance (English-centric, no orientation / unwarp for speed)
OCR = PaddleOCR(
    use_angle_cls=False,
    use_gpu=False,
    lang="en",
    use_doc_unwarping=False,
    use_textline_orientation=False
)

app = FastAPI(title="PaddleOCR Server", version="1.0")

@app.get("/healthz")
async def healthz():
    return {"ok": True}

class OCRRequest(BaseModel):
    image_url: Optional[str] = None

def _run_ocr(img_bytes: bytes):
    # PaddleOCR can accept numpy arrays / file paths; we'll use bytes -> PIL -> numpy
    im = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    # Save to a temp in-memory buffer to ensure friendly format
    buf = io.BytesIO()
    im.save(buf, format="PNG")
    buf.seek(0)
    # OCR.predict can accept bytes-like path? We'll pass the PIL image's array:
    import numpy as np
    arr = np.array(im)
    result = OCR.ocr(arr, cls=False)
    out = []
    # result is list per page; we only pass one image
    for line in result[0]:
        ((x0,y0),(x1,y1),(x2,y2),(x3,y3)), (text, conf) = line
        out.append({
            "text": text,
            "confidence": float(conf),
            "box": [[float(x0),float(y0)],[float(x1),float(y1)],
                    [float(x2),float(y2)],[float(x3),float(y3)]]
        })
    return out

@app.post("/ocr")
async def ocr_endpoint(
    file: UploadFile = File(None),
    image_url: Optional[str] = Form(None)
):
    try:
        data = None
        if file is not None:
            data = await file.read()
        elif image_url:
            timeout = httpx.Timeout(15.0, connect=5.0)
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                r = await client.get(image_url)
                if r.status_code != 200:
                    raise HTTPException(status_code=400, detail=f"Failed to fetch image: {r.status_code}")
                data = r.content
        else:
            raise HTTPException(status_code=400, detail="Provide multipart 'file' or 'image_url'")

        # Optional: downscale super-large images for speed
        if len(data) > 5_000_000:  # ~5MB heuristic
            im = Image.open(io.BytesIO(data)).convert("RGB")
            max_side = 2000
            scale = min(1.0, max_side / max(im.size))
            if scale < 1.0:
                im = im.resize((int(im.width*scale), int(im.height*scale)))
                buf = io.BytesIO()
                im.save(buf, format="JPEG", quality=92)
                data = buf.getvalue()

        results = await asyncio.to_thread(_run_ocr, data)
        return JSONResponse({"results": results})
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
