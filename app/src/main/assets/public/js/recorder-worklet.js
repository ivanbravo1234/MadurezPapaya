// AudioWorklet que entrega al hilo principal las muestras crudas del
// micrófono (canal 0) en bloques, sin ningún procesamiento.
class RecorderProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (ch) this.port.postMessage(ch.slice(0));
    return true;
  }
}
registerProcessor("recorder-processor", RecorderProcessor);
