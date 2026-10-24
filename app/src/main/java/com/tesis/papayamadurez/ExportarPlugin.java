package com.tesis.papayamadurez;

import android.content.Intent;
import android.net.Uri;
import androidx.core.content.FileProvider;
import com.getcapacitor.Plugin;
import com.getcapacitor.PluginCall;
import com.getcapacitor.PluginMethod;
import com.getcapacitor.annotation.CapacitorPlugin;
import java.io.File;
import java.io.FileOutputStream;
import java.io.OutputStreamWriter;
import java.io.Writer;
import java.nio.charset.StandardCharsets;

/**
 * Exporta un archivo de texto generado por la app (el historial en CSV) y lo
 * abre en el menú Compartir de Android: Drive, correo, WhatsApp, Archivos...
 * El WebView no puede descargar archivos por sí mismo, por eso se hace aquí.
 *
 * Desde JavaScript: Capacitor.nativePromise("Exportar", "compartir",
 *                     { nombre, contenido, tipo })
 */
@CapacitorPlugin(name = "Exportar")
public class ExportarPlugin extends Plugin {

    @PluginMethod
    public void compartir(PluginCall call) {
        String nombre = call.getString("nombre", "historial.csv").replaceAll("[^A-Za-z0-9._-]", "_");
        String contenido = call.getString("contenido", "");
        String tipo = call.getString("tipo", "text/csv");
        try {
            File carpeta = new File(getContext().getCacheDir(), "exportar");
            if (!carpeta.exists() && !carpeta.mkdirs()) {
                throw new Exception("no se pudo crear la carpeta temporal");
            }
            File archivo = new File(carpeta, nombre);
            try (Writer w = new OutputStreamWriter(new FileOutputStream(archivo), StandardCharsets.UTF_8)) {
                w.write('﻿'); // BOM: así Excel muestra bien las tildes
                w.write(contenido);
            }
            Uri uri = FileProvider.getUriForFile(getContext(), getContext().getPackageName() + ".fileprovider", archivo);
            Intent enviar = new Intent(Intent.ACTION_SEND);
            enviar.setType(tipo);
            enviar.putExtra(Intent.EXTRA_STREAM, uri);
            enviar.putExtra(Intent.EXTRA_SUBJECT, nombre);
            enviar.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            Intent menu = Intent.createChooser(enviar, "Exportar historial");
            menu.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION);
            getActivity().startActivity(menu);
            call.resolve();
        } catch (Exception e) {
            call.reject("No se pudo exportar: " + e.getMessage());
        }
    }
}
