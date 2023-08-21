package com.baidu.paddle.fastdeploy.app.ui.view;

import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.SurfaceTexture;
import android.hardware.Camera;
import android.hardware.Camera.CameraInfo;
import android.hardware.Camera.Size;
import android.opengl.GLES11Ext;
import android.opengl.GLES20;
import android.opengl.GLSurfaceView;
import android.opengl.GLSurfaceView.Renderer;
import android.opengl.GLUtils;
import android.opengl.Matrix;
import android.os.SystemClock;
import android.util.AttributeSet;
import android.util.Log;

import com.baidu.paddle.fastdeploy.app.ui.Utils;

import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.util.List;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

public class CameraSurfaceView extends GLSurfaceView implements Renderer,
        SurfaceTexture.OnFrameAvailableListener {
    private static final String TAG = CameraSurfaceView.class.getSimpleName();

    public static int EXPECTED_PREVIEW_WIDTH = 1280;  // 1920
    public static int EXPECTED_PREVIEW_HEIGHT = 720;  // 960

    protected int numberOfCameras;
    protected int selectedCameraId;
    protected boolean disableCamera = false;
    protected Camera camera;

    protected Context context;
    protected SurfaceTexture surfaceTexture;
    protected int surfaceWidth = 0;
    protected int surfaceHeight = 0;
    protected int textureWidth = 0;
    protected int textureHeight = 0;

    protected Bitmap ARGB8888ImageBitmap;
    protected boolean bitmapReleaseMode = true;

    // In order to manipulate the camera preview data and render the modified one
    // to the screen, three textures are created and the data flow is shown as following:
    // previewdata->camTextureId->fboTexureId->drawTexureId->framebuffer
    protected int[] fbo = {0};
    protected int[] camTextureId = {0};
    protected int[] fboTexureId = {0};
    protected int[] drawTexureId = {0};

    private final String vss = ""
            + "attribute vec2 vPosition;\n"
            + "attribute vec2 vTexCoord;\n" + "varying vec2 texCoord;\n"
            + "void main() {\n" + "  texCoord = vTexCoord;\n"
            + "  gl_Position = vec4 (vPosition.x, vPosition.y, 0.0, 1.0);\n"
            + "}";

    private final String fssCam2FBO = ""
            + "#extension GL_OES_EGL_image_external : require\n"
            + "precision mediump float;\n"
            + "uniform samplerExternalOES sTexture;\n"
            + "varying vec2 texCoord;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(sTexture,texCoord);\n" + "}";

    private final String fssTex2Screen = ""
            + "precision mediump float;\n"
            + "uniform sampler2D sTexture;\n"
            + "varying vec2 texCoord;\n"
            + "void main() {\n"
            + "  gl_FragColor = texture2D(sTexture,texCoord);\n" + "}";

    private final float[] vertexCoords = {
            -1, -1,
            -1, 1,
            1, -1,
            1, 1};
    private float[] textureCoords = {
            0, 1,
            0, 0,
            1, 1,
            1, 0};

    private FloatBuffer vertexCoordsBuffer;
    private FloatBuffer textureCoordsBuffer;

    private int progCam2FBO = -1;
    private int progTex2Screen = -1;
    private int vcCam2FBO;
    private int tcCam2FBO;
    private int vcTex2Screen;
    private int tcTex2Screen;

    public void setBitmapReleaseMode(boolean mode) {
        synchronized (this) {
            bitmapReleaseMode = mode;
        }
    }

    public Bitmap getBitmap() {
        return ARGB8888ImageBitmap; // may null or recycled.
    }

    public interface OnTextureChangedListener {
        boolean onTextureChanged(Bitmap ARGB8888ImageBitmap);
    }

    private OnTextureChangedListener onTextureChangedListener = null;

    public void setOnTextureChangedListener(OnTextureChangedListener listener) {
        onTextureChangedListener = listener;
    }

    public CameraSurfaceView(Context ctx, AttributeSet attrs) {
        super(ctx, attrs);
        context = ctx;
        setEGLContextClientVersion(2);
        setRenderer(this);
        setRenderMode(RENDERMODE_WHEN_DIRTY);

        // Find the total number of available cameras and the ID of the default camera
        numberOfCameras = Camera.getNumberOfCameras();
        CameraInfo cameraInfo = new CameraInfo();
        for (int i = 0; i < numberOfCameras; i++) {
            Camera.getCameraInfo(i, cameraInfo);
            if (cameraInfo.facing == CameraInfo.CAMERA_FACING_BACK) {
                selectedCameraId = i;
            }
        }
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        // Create OES texture for storing camera preview data(YUV format)
        GLES20.glGenTextures(1, camTextureId, 0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, camTextureId[0]);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        surfaceTexture = new SurfaceTexture(camTextureId[0]);
        surfaceTexture.setOnFrameAvailableListener(this);

        // Prepare vertex and texture coordinates
        int bytes = vertexCoords.length * Float.SIZE / Byte.SIZE;
        vertexCoordsBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        textureCoordsBuffer = ByteBuffer.allocateDirect(bytes).order(ByteOrder.nativeOrder()).asFloatBuffer();
        vertexCoordsBuffer.put(vertexCoords).position(0);
        textureCoordsBuffer.put(textureCoords).position(0);

        // Create vertex and fragment shaders
        // camTextureId->fboTexureId
        progCam2FBO = Utils.createShaderProgram(vss, fssCam2FBO);
        vcCam2FBO = GLES20.glGetAttribLocation(progCam2FBO, "vPosition");
        tcCam2FBO = GLES20.glGetAttribLocation(progCam2FBO, "vTexCoord");
        GLES20.glEnableVertexAttribArray(vcCam2FBO);
        GLES20.glEnableVertexAttribArray(tcCam2FBO);
        // fboTexureId/drawTexureId -> screen
        progTex2Screen = Utils.createShaderProgram(vss, fssTex2Screen);
        vcTex2Screen = GLES20.glGetAttribLocation(progTex2Screen, "vPosition");
        tcTex2Screen = GLES20.glGetAttribLocation(progTex2Screen, "vTexCoord");
        GLES20.glEnableVertexAttribArray(vcTex2Screen);
        GLES20.glEnableVertexAttribArray(tcTex2Screen);
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        surfaceWidth = width;
        surfaceHeight = height;
        openCamera();
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        if (surfaceTexture == null) return;

        GLES20.glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT | GLES20.GL_DEPTH_BUFFER_BIT);
        surfaceTexture.updateTexImage();
        float[] matrix = new float[16];
        surfaceTexture.getTransformMatrix(matrix);

        // camTextureId->fboTexureId
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0]);
        GLES20.glViewport(0, 0, textureWidth, textureHeight);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glUseProgram(progCam2FBO);
        GLES20.glVertexAttribPointer(vcCam2FBO, 2, GLES20.GL_FLOAT, false, 4 * 2, vertexCoordsBuffer);
        textureCoordsBuffer.clear();
        textureCoordsBuffer.put(transformTextureCoordinates(textureCoords, matrix));
        textureCoordsBuffer.position(0);
        GLES20.glVertexAttribPointer(tcCam2FBO, 2, GLES20.GL_FLOAT, false, 4 * 2, textureCoordsBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, camTextureId[0]);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(progCam2FBO, "sTexture"), 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glFlush();

        // Check if the draw texture is set
        int targetTexureId = fboTexureId[0];
        if (onTextureChangedListener != null) {
            // Read pixels of FBO to a bitmap
            ByteBuffer pixelBuffer = ByteBuffer.allocate(textureWidth * textureHeight * 4);
            GLES20.glReadPixels(0, 0, textureWidth, textureHeight, GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, pixelBuffer);

            ARGB8888ImageBitmap = Bitmap.createBitmap(textureWidth, textureHeight, Bitmap.Config.ARGB_8888);
            ARGB8888ImageBitmap.copyPixelsFromBuffer(pixelBuffer);

            boolean modified = onTextureChangedListener.onTextureChanged(ARGB8888ImageBitmap);

            if (modified) {
                targetTexureId = drawTexureId[0];
                // Update a bitmap to the GL texture if modified
                GLES20.glActiveTexture(targetTexureId);
                // GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, targetTexureId);
                GLES20.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, targetTexureId);
                GLUtils.texImage2D(GL10.GL_TEXTURE_2D, 0, ARGB8888ImageBitmap, 0);
            }
            if (bitmapReleaseMode) {
                ARGB8888ImageBitmap.recycle();
            }
        }

        // fboTexureId/drawTexureId->Screen
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glViewport(0, 0, surfaceWidth, surfaceHeight);
        GLES20.glClear(GLES20.GL_COLOR_BUFFER_BIT);
        GLES20.glUseProgram(progTex2Screen);
        GLES20.glVertexAttribPointer(vcTex2Screen, 2, GLES20.GL_FLOAT, false, 4 * 2, vertexCoordsBuffer);
        textureCoordsBuffer.clear();
        textureCoordsBuffer.put(textureCoords);
        textureCoordsBuffer.position(0);
        GLES20.glVertexAttribPointer(tcTex2Screen, 2, GLES20.GL_FLOAT, false, 4 * 2, textureCoordsBuffer);
        GLES20.glActiveTexture(GLES20.GL_TEXTURE0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, targetTexureId);
        GLES20.glUniform1i(GLES20.glGetUniformLocation(progTex2Screen, "sTexture"), 0);
        GLES20.glDrawArrays(GLES20.GL_TRIANGLE_STRIP, 0, 4);
        GLES20.glFlush();
    }

    private float[] transformTextureCoordinates(float[] coords, float[] matrix) {
        float[] result = new float[coords.length];
        float[] vt = new float[4];
        for (int i = 0; i < coords.length; i += 2) {
            float[] v = {coords[i], coords[i + 1], 0, 1};
            Matrix.multiplyMV(vt, 0, matrix, 0, v, 0);
            result[i] = vt[0];
            result[i + 1] = vt[1];
        }
        return result;
    }

    @Override
    public void onResume() {
        super.onResume();
    }

    @Override
    public void onPause() {
        super.onPause();
        releaseCamera();
    }

    @Override
    public void onFrameAvailable(SurfaceTexture surfaceTexture) {
        requestRender();
    }

    public void disableCamera() {
        disableCamera = true;
    }

    public void enableCamera() {
        disableCamera = false;
    }

    public void switchCamera() {
        releaseCamera();
        selectedCameraId = (selectedCameraId + 1) % numberOfCameras;
        openCamera();
    }

    public void openCamera() {
        if (disableCamera) return;
        camera = Camera.open(selectedCameraId);
        List<Size> supportedPreviewSizes = camera.getParameters().getSupportedPreviewSizes();
        Size previewSize = Utils.getOptimalPreviewSize(supportedPreviewSizes, EXPECTED_PREVIEW_WIDTH,
                EXPECTED_PREVIEW_HEIGHT);
        Camera.Parameters parameters = camera.getParameters();
        parameters.setPreviewSize(previewSize.width, previewSize.height);
        if (parameters.getSupportedFocusModes().contains(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO)) {
            parameters.setFocusMode(Camera.Parameters.FOCUS_MODE_CONTINUOUS_VIDEO);
        }
        camera.setParameters(parameters);
        int degree = Utils.getCameraDisplayOrientation(context, selectedCameraId);
        camera.setDisplayOrientation(degree);
        boolean rotate = degree == 90 || degree == 270;
        textureWidth = rotate ? previewSize.height : previewSize.width;
        textureHeight = rotate ? previewSize.width : previewSize.height;
        // Destroy FBO and draw textures
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, 0);
        GLES20.glDeleteFramebuffers(1, fbo, 0);
        GLES20.glDeleteTextures(1, drawTexureId, 0);
        GLES20.glDeleteTextures(1, fboTexureId, 0);
        // Normal texture for storing modified camera preview data(RGBA format)
        GLES20.glGenTextures(1, drawTexureId, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, drawTexureId[0]);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, textureWidth, textureHeight, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        // FBO texture for storing camera preview data(RGBA format)
        GLES20.glGenTextures(1, fboTexureId, 0);
        GLES20.glBindTexture(GLES20.GL_TEXTURE_2D, fboTexureId[0]);
        GLES20.glTexImage2D(GLES20.GL_TEXTURE_2D, 0, GLES20.GL_RGBA, textureWidth, textureHeight, 0,
                GLES20.GL_RGBA, GLES20.GL_UNSIGNED_BYTE, null);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_S, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_WRAP_T, GLES20.GL_CLAMP_TO_EDGE);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MIN_FILTER, GLES20.GL_NEAREST);
        GLES20.glTexParameteri(GLES20.GL_TEXTURE_2D, GLES20.GL_TEXTURE_MAG_FILTER, GLES20.GL_NEAREST);
        // Generate FBO and bind to FBO texture
        GLES20.glGenFramebuffers(1, fbo, 0);
        GLES20.glBindFramebuffer(GLES20.GL_FRAMEBUFFER, fbo[0]);
        GLES20.glFramebufferTexture2D(GLES20.GL_FRAMEBUFFER, GLES20.GL_COLOR_ATTACHMENT0, GLES20.GL_TEXTURE_2D,
                fboTexureId[0], 0);
        try {
            camera.setPreviewTexture(surfaceTexture);
        } catch (IOException exception) {
            Log.e(TAG, "IOException caused by setPreviewDisplay()", exception);
        }
        camera.startPreview();
    }

    public void releaseCamera() {
        if (camera != null) {
            camera.setPreviewCallback(null);
            camera.stopPreview();
            camera.release();
            camera = null;
        }
    }
}
