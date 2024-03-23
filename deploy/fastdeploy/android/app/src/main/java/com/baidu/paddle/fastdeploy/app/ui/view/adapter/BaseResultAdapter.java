package com.baidu.paddle.fastdeploy.app.ui.view.adapter;

import android.content.Context;
import android.support.annotation.NonNull;
import android.support.annotation.Nullable;
import android.view.LayoutInflater;
import android.view.View;
import android.view.ViewGroup;
import android.widget.ArrayAdapter;
import android.widget.TextView;

import com.baidu.paddle.fastdeploy.app.examples.R;
import com.baidu.paddle.fastdeploy.app.ui.view.model.BaseResultModel;

import java.text.DecimalFormat;
import java.util.List;

public class BaseResultAdapter extends ArrayAdapter<BaseResultModel> {
    private int resourceId;

    public BaseResultAdapter(@NonNull Context context, int resource) {
        super(context, resource);
    }

    public BaseResultAdapter(@NonNull Context context, int resource, @NonNull List<BaseResultModel> objects) {
        super(context, resource, objects);
        resourceId = resource;
    }

    @NonNull
    @Override
    public View getView(int position, @Nullable View convertView, @NonNull ViewGroup parent) {
        BaseResultModel model = getItem(position);
        View view = LayoutInflater.from(getContext()).inflate(resourceId, null);
        TextView indexText = (TextView) view.findViewById(R.id.index);
        TextView nameText = (TextView) view.findViewById(R.id.name);
        TextView confidenceText = (TextView) view.findViewById(R.id.confidence);
        indexText.setText(String.valueOf(model.getIndex()));
        nameText.setText(String.valueOf(model.getName()));
        confidenceText.setText(formatFloatString(model.getConfidence()));
        return view;
    }

    public static String formatFloatString(float number) {
        DecimalFormat df = new DecimalFormat("0.00");
        return df.format(number);
    }
}
