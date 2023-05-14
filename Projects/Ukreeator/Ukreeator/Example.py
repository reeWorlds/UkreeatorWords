# -*- coding: windows-1251 -*-

import Ukreeator


def showExamples(generator, model_type):
	print("Generate " + model_type)
	for model_size in ["tiny", "normal", "extra"]:
		generator.load_model(model_size, "../../../models/" + model_type + "/" + model_size + ".pth")
		
		print(f"Here are 5 {model_type} generated by " + model_size + " model")
		print(" ".join(generator.generateWord() for i in range(5)))

	print(f"Here are 5 extra {model_type} that start with \"��\"")
	print(" ".join(generator.generateWord("��") for i in range(5)))


if __name__ == "__main__":
	generator = Ukreeator.UkreeatorWords()
	generator.set_torch_seed(74)

	showExamples(generator, "names")
	print()

	showExamples(generator, "locations")
	print()

	showExamples(generator, "words")
	print()