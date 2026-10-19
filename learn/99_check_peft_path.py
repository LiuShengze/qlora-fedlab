import inspect
import peft

print(peft.__file__)                          # peft 库装在哪
print(inspect.getsourcefile(type(peft.config.LoraConfig("CAUSAL_LM"))))
