package com.baidu.paddle.lite.demo.ocr;

import android.content.SharedPreferences;
import android.os.Bundle;
import android.preference.CheckBoxPreference;
import android.preference.EditTextPreference;
import android.preference.ListPreference;

import androidx.appcompat.app.ActionBar;

import java.util.ArrayList;
import java.util.List;


public class SettingsActivity extends AppCompatPreferenceActivity implements SharedPreferences.OnSharedPreferenceChangeListener {
    ListPreference lpChoosePreInstalledModel = null;
    CheckBoxPreference cbEnableCustomSettings = null;
    EditTextPreference etModelPath = null;
    EditTextPreference etLabelPath = null;
    ListPreference etImagePath = null;
    ListPreference lpCPUThreadNum = null;
    ListPreference lpCPUPowerMode = null;
    EditTextPreference etDetLongSize = null;
    EditTextPreference etScoreThreshold = null;

    List<String> preInstalledModelPaths = null;
    List<String> preInstalledLabelPaths = null;
    List<String> preInstalledImagePaths = null;
    List<String> preInstalledDetLongSizes = null;
    List<String> preInstalledCPUThreadNums = null;
    List<String> preInstalledCPUPowerModes = null;
    List<String> preInstalledInputColorFormats = null;
    List<String> preInstalledInputMeans = null;
    List<String> preInstalledInputStds = null;
    List<String> preInstalledScoreThresholds = null;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        addPreferencesFromResource(R.xml.settings);
        ActionBar supportActionBar = getSupportActionBar();
        if (supportActionBar != null) {
            supportActionBar.setDisplayHomeAsUpEnabled(true);
        }

        // Initialized pre-installed models
        preInstalledModelPaths = new ArrayList<String>();
        preInstalledLabelPaths = new ArrayList<String>();
        preInstalledImagePaths = new ArrayList<String>();
        preInstalledDetLongSizes = new ArrayList<String>();
        preInstalledCPUThreadNums = new ArrayList<String>();
        preInstalledCPUPowerModes = new ArrayList<String>();
        preInstalledInputColorFormats = new ArrayList<String>();
        preInstalledInputMeans = new ArrayList<String>();
        preInstalledInputStds = new ArrayList<String>();
        preInstalledScoreThresholds = new ArrayList<String>();
        // Add ssd_mobilenet_v1_pascalvoc_for_cpu
        preInstalledModelPaths.add(getString(R.string.MODEL_PATH_DEFAULT));
        preInstalledLabelPaths.add(getString(R.string.LABEL_PATH_DEFAULT));
        preInstalledImagePaths.add(getString(R.string.IMAGE_PATH_DEFAULT));
        preInstalledCPUThreadNums.add(getString(R.string.CPU_THREAD_NUM_DEFAULT));
        preInstalledCPUPowerModes.add(getString(R.string.CPU_POWER_MODE_DEFAULT));
        preInstalledDetLongSizes.add(getString(R.string.DET_LONG_SIZE_DEFAULT));
        preInstalledScoreThresholds.add(getString(R.string.SCORE_THRESHOLD_DEFAULT));

        // Setup UI components
        lpChoosePreInstalledModel =
                (ListPreference) findPreference(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY));
        String[] preInstalledModelNames = new String[preInstalledModelPaths.size()];
        for (int i = 0; i < preInstalledModelPaths.size(); i++) {
            preInstalledModelNames[i] =
                    preInstalledModelPaths.get(i).substring(preInstalledModelPaths.get(i).lastIndexOf("/") + 1);
        }
        lpChoosePreInstalledModel.setEntries(preInstalledModelNames);
        lpChoosePreInstalledModel.setEntryValues(preInstalledModelPaths.toArray(new String[preInstalledModelPaths.size()]));
        cbEnableCustomSettings =
                (CheckBoxPreference) findPreference(getString(R.string.ENABLE_CUSTOM_SETTINGS_KEY));
        etModelPath = (EditTextPreference) findPreference(getString(R.string.MODEL_PATH_KEY));
        etModelPath.setTitle("Model Path (SDCard: " + Utils.getSDCardDirectory() + ")");
        etLabelPath = (EditTextPreference) findPreference(getString(R.string.LABEL_PATH_KEY));
        etImagePath = (ListPreference) findPreference(getString(R.string.IMAGE_PATH_KEY));
        lpCPUThreadNum =
                (ListPreference) findPreference(getString(R.string.CPU_THREAD_NUM_KEY));
        lpCPUPowerMode =
                (ListPreference) findPreference(getString(R.string.CPU_POWER_MODE_KEY));
        etDetLongSize = (EditTextPreference) findPreference(getString(R.string.DET_LONG_SIZE_KEY));
        etScoreThreshold = (EditTextPreference) findPreference(getString(R.string.SCORE_THRESHOLD_KEY));
    }

    private void reloadPreferenceAndUpdateUI() {
        SharedPreferences sharedPreferences = getPreferenceScreen().getSharedPreferences();
        boolean enableCustomSettings =
                sharedPreferences.getBoolean(getString(R.string.ENABLE_CUSTOM_SETTINGS_KEY), false);
        String modelPath = sharedPreferences.getString(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        int modelIdx = lpChoosePreInstalledModel.findIndexOfValue(modelPath);
        if (modelIdx >= 0 && modelIdx < preInstalledModelPaths.size()) {
            if (!enableCustomSettings) {
                SharedPreferences.Editor editor = sharedPreferences.edit();
                editor.putString(getString(R.string.MODEL_PATH_KEY), preInstalledModelPaths.get(modelIdx));
                editor.putString(getString(R.string.LABEL_PATH_KEY), preInstalledLabelPaths.get(modelIdx));
                editor.putString(getString(R.string.IMAGE_PATH_KEY), preInstalledImagePaths.get(modelIdx));
                editor.putString(getString(R.string.CPU_THREAD_NUM_KEY), preInstalledCPUThreadNums.get(modelIdx));
                editor.putString(getString(R.string.CPU_POWER_MODE_KEY), preInstalledCPUPowerModes.get(modelIdx));
                editor.putString(getString(R.string.DET_LONG_SIZE_KEY), preInstalledDetLongSizes.get(modelIdx));
                editor.putString(getString(R.string.SCORE_THRESHOLD_KEY),
                        preInstalledScoreThresholds.get(modelIdx));
                editor.apply();
            }
            lpChoosePreInstalledModel.setSummary(modelPath);
        }
        cbEnableCustomSettings.setChecked(enableCustomSettings);
        etModelPath.setEnabled(enableCustomSettings);
        etLabelPath.setEnabled(enableCustomSettings);
        etImagePath.setEnabled(enableCustomSettings);
        lpCPUThreadNum.setEnabled(enableCustomSettings);
        lpCPUPowerMode.setEnabled(enableCustomSettings);
        etDetLongSize.setEnabled(enableCustomSettings);
        etScoreThreshold.setEnabled(enableCustomSettings);
        modelPath = sharedPreferences.getString(getString(R.string.MODEL_PATH_KEY),
                getString(R.string.MODEL_PATH_DEFAULT));
        String labelPath = sharedPreferences.getString(getString(R.string.LABEL_PATH_KEY),
                getString(R.string.LABEL_PATH_DEFAULT));
        String imagePath = sharedPreferences.getString(getString(R.string.IMAGE_PATH_KEY),
                getString(R.string.IMAGE_PATH_DEFAULT));
        String cpuThreadNum = sharedPreferences.getString(getString(R.string.CPU_THREAD_NUM_KEY),
                getString(R.string.CPU_THREAD_NUM_DEFAULT));
        String cpuPowerMode = sharedPreferences.getString(getString(R.string.CPU_POWER_MODE_KEY),
                getString(R.string.CPU_POWER_MODE_DEFAULT));
        String detLongSize = sharedPreferences.getString(getString(R.string.DET_LONG_SIZE_KEY),
                getString(R.string.DET_LONG_SIZE_DEFAULT));
        String scoreThreshold = sharedPreferences.getString(getString(R.string.SCORE_THRESHOLD_KEY),
                getString(R.string.SCORE_THRESHOLD_DEFAULT));
        etModelPath.setSummary(modelPath);
        etModelPath.setText(modelPath);
        etLabelPath.setSummary(labelPath);
        etLabelPath.setText(labelPath);
        etImagePath.setSummary(imagePath);
        etImagePath.setValue(imagePath);
        lpCPUThreadNum.setValue(cpuThreadNum);
        lpCPUThreadNum.setSummary(cpuThreadNum);
        lpCPUPowerMode.setValue(cpuPowerMode);
        lpCPUPowerMode.setSummary(cpuPowerMode);
        etDetLongSize.setSummary(detLongSize);
        etDetLongSize.setText(detLongSize);
        etScoreThreshold.setText(scoreThreshold);
        etScoreThreshold.setSummary(scoreThreshold);
    }

    @Override
    protected void onResume() {
        super.onResume();
        getPreferenceScreen().getSharedPreferences().registerOnSharedPreferenceChangeListener(this);
        reloadPreferenceAndUpdateUI();
    }

    @Override
    protected void onPause() {
        super.onPause();
        getPreferenceScreen().getSharedPreferences().unregisterOnSharedPreferenceChangeListener(this);
    }

    @Override
    public void onSharedPreferenceChanged(SharedPreferences sharedPreferences, String key) {
        if (key.equals(getString(R.string.CHOOSE_PRE_INSTALLED_MODEL_KEY))) {
            SharedPreferences.Editor editor = sharedPreferences.edit();
            editor.putBoolean(getString(R.string.ENABLE_CUSTOM_SETTINGS_KEY), false);
            editor.commit();
        }
        reloadPreferenceAndUpdateUI();
    }
}
