package com.baidu.paddle.lite.demo.ocr;

import android.content.Context;
import android.graphics.Bitmap;
import android.os.Environment;

import java.io.*;

public class Utils {
    private static final String TAG = Utils.class.getSimpleName();

    public static void copyFileFromAssets(Context appCtx, String srcPath, String dstPath) {
        if (srcPath.isEmpty() || dstPath.isEmpty()) {
            return;
        }
        InputStream is = null;
        OutputStream os = null;
        try {
            is = new BufferedInputStream(appCtx.getAssets().open(srcPath));
            os = new BufferedOutputStream(new FileOutputStream(new File(dstPath)));
            byte[] buffer = new byte[1024];
            int length = 0;
            while ((length = is.read(buffer)) != -1) {
                os.write(buffer, 0, length);
            }
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        } finally {
            try {
                os.close();
                is.close();
            } catch (IOException e) {
                e.printStackTrace();
            }
        }
    }

    public static void copyDirectoryFromAssets(Context appCtx, String srcDir, String dstDir) {
        if (srcDir.isEmpty() || dstDir.isEmpty()) {
            return;
        }
        try {
            if (!new File(dstDir).exists()) {
                new File(dstDir).mkdirs();
            }
            for (String fileName : appCtx.getAssets().list(srcDir)) {
                String srcSubPath = srcDir + File.separator + fileName;
                String dstSubPath = dstDir + File.separator + fileName;
                if (new File(srcSubPath).isDirectory()) {
                    copyDirectoryFromAssets(appCtx, srcSubPath, dstSubPath);
                } else {
                    copyFileFromAssets(appCtx, srcSubPath, dstSubPath);
                }
            }
        } catch (Exception e) {
            e.printStackTrace();
        }
    }

    public static float[] parseFloatsFromString(String string, String delimiter) {
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        float[] floats = new float[pieces.length];
        for (int i = 0; i < pieces.length; i++) {
            floats[i] = Float.parseFloat(pieces[i].trim());
        }
        return floats;
    }

    public static long[] parseLongsFromString(String string, String delimiter) {
        String[] pieces = string.trim().toLowerCase().split(delimiter);
        long[] longs = new long[pieces.length];
        for (int i = 0; i < pieces.length; i++) {
            longs[i] = Long.parseLong(pieces[i].trim());
        }
        return longs;
    }

    public static String getSDCardDirectory() {
        return Environment.getExternalStorageDirectory().getAbsolutePath();
    }

    public static boolean isSupportedNPU() {
        return false;
        // String hardware = android.os.Build.HARDWARE;
        // return hardware.equalsIgnoreCase("kirin810") || hardware.equalsIgnoreCase("kirin990");
    }

    public static Bitmap resizeWithStep(Bitmap bitmap, int maxLength, int step) {
        int width = bitmap.getWidth();
        int height = bitmap.getHeight();
        int maxWH = Math.max(width, height);
        float ratio = 1;
        int newWidth = width;
        int newHeight = height;
        if (maxWH > maxLength) {
            ratio = maxLength * 1.0f / maxWH;
            newWidth = (int) Math.floor(ratio * width);
            newHeight = (int) Math.floor(ratio * height);
        }

        newWidth = newWidth - newWidth % step;
        if (newWidth == 0) {
            newWidth = step;
        }
        newHeight = newHeight - newHeight % step;
        if (newHeight == 0) {
            newHeight = step;
        }
        return Bitmap.createScaledBitmap(bitmap, newWidth, newHeight, true);
    }
}
