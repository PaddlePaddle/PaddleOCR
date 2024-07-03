package com.baidu.paddle.fastdeploy.app.ui.view;

import android.content.Context;
import android.os.Handler;
import android.util.AttributeSet;
import android.widget.ListView;

public class ResultListView extends ListView {
    public ResultListView(Context context) {
        super(context);
    }

    public ResultListView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public ResultListView(Context context, AttributeSet attrs, int defStyleAttr) {
        super(context, attrs, defStyleAttr);
    }

    private Handler handler;

    public void setHandler(Handler mHandler) {
        handler = mHandler;
    }

    public void clear() {
        handler.post(new Runnable() {
            @Override
            public void run() {
                removeAllViewsInLayout();
                invalidate();
            }
        });
    }

    @Override
    protected void onMeasure(int widthMeasureSpec, int heightMeasureSpec) {
        int expandSpec = MeasureSpec.makeMeasureSpec(Integer.MAX_VALUE >> 2,
                MeasureSpec.AT_MOST);
        super.onMeasure(widthMeasureSpec, expandSpec);
    }
}
