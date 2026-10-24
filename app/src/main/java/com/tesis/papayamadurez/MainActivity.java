package com.tesis.papayamadurez;

import android.os.Bundle;
import com.getcapacitor.BridgeActivity;

public class MainActivity extends BridgeActivity {

    @Override
    public void onCreate(Bundle savedInstanceState) {
        // Plugins propios: se registran antes de crear el puente.
        registerPlugin(ExportarPlugin.class);
        super.onCreate(savedInstanceState);
    }
}
