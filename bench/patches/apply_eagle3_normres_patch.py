"""SGLang 0.5.22 LlamaForCausalLMEagle3 computes residual = raw target hidden (norm AFTER residual capture).
RedHatAI speculators EAGLE-3 checkpoints are trained with norm_before_residual=True (residual = normed hidden).
This patch makes llama_eagle3.py honour config.norm_before_residual. Idempotent. --revert supported."""
import sys, os, sglang
f = os.path.dirname(sglang.__file__) + "/srt/models/llama_eagle3.py"
src = open(f).read()
if "--revert" in sys.argv:
    os.replace(f + ".orig", f); print("reverted"); sys.exit()
if "NORM_BEFORE_RES_PATCH" in src: print("already"); sys.exit()
old = "            residual = hidden_states\n            hidden_states = self.hidden_norm(hidden_states)\n            embeds = self.input_layernorm(embeds)\n"
new = ("            if getattr(self, 'norm_before_residual', False):  # NORM_BEFORE_RES_PATCH\n"
       "                hidden_states = self.hidden_norm(hidden_states)\n"
       "                residual = hidden_states\n"
       "            else:\n"
       "                residual = hidden_states\n"
       "                hidden_states = self.hidden_norm(hidden_states)\n"
       "            embeds = self.input_layernorm(embeds)\n")
assert src.count(old) == 1
src = src.replace(old, new)
old2 = "        self.hidden_norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)\n"
assert src.count(old2) == 1
src = src.replace(old2, old2 + "        self.norm_before_residual = bool(getattr(config, 'norm_before_residual', False))\n")
open(f + ".orig", "w").write(open(f).read()); open(f, "w").write(src); print("patched", f)
