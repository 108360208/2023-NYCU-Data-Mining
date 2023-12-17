	@echo on


	python ItemsetVerifier.py -r ../../../Apriori_python/dataset/C/step2_task1_dataset(C)_0.01_result1.txt -s 0.01 -s ../../../Apriori_python/dataset/negFIN_C/step3_task1_dataset(C)_0.01_result1.txt
	python ItemsetVerifier.py -r ../../../Apriori_python/dataset/C/step2_task1_dataset(C)_0.02_result1.txt -s 0.02 -s ../../../Apriori_python/dataset/negFIN_C/step3_task1_dataset(C)_0.02_result1.txt
	python ItemsetVerifier.py -r ../../../Apriori_python/dataset/C/step2_task1_dataset(C)_0.03_result1.txt -s 0.03 -s ../../../Apriori_python/dataset/negFIN_C/step3_task1_dataset(C)_0.03_result1.txt

	pause	