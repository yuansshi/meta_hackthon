// Mic → 16 kHz mono PCM16 in 20 ms frames, posted to the main thread with a level for the meter.
// `sampleRate` is the AudioContext rate (a global in worklet scope), usually 44.1 or 48 kHz.
class Pcm16k extends AudioWorkletProcessor {
  constructor() {
    super();
    this.ratio = sampleRate / 16000;
    this.alpha = 1 - Math.exp((-2 * Math.PI * 7000) / sampleRate); // one-pole low-pass before decimating
    this.lp = 0;
    this.prev = 0; // last filtered sample of the previous block (index -1)
    this.pos = -1; // fractional read position relative to the current block
    this.out = new Int16Array(320);
    this.n = 0;
    this.sumSq = 0;
  }

  push(v) {
    const s = Math.max(-1, Math.min(1, v));
    this.sumSq += s * s;
    this.out[this.n++] = s < 0 ? s * 0x8000 : s * 0x7fff;
    if (this.n === this.out.length) {
      const pcm = this.out.buffer;
      this.port.postMessage({ pcm, level: Math.sqrt(this.sumSq / this.n) }, [pcm]);
      this.out = new Int16Array(320);
      this.n = 0;
      this.sumSq = 0;
    }
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    const x = new Float32Array(ch.length);
    for (let i = 0; i < ch.length; i++) x[i] = this.lp += this.alpha * (ch[i] - this.lp);
    let p = this.pos;
    while (p < x.length - 1) {
      const i = Math.floor(p);
      const f = p - i;
      const s0 = i < 0 ? this.prev : x[i];
      this.push(s0 + (x[i + 1] - s0) * f);
      p += this.ratio;
    }
    this.pos = p - x.length;
    this.prev = x[x.length - 1];
    return true;
  }
}

registerProcessor("pcm16k", Pcm16k);
