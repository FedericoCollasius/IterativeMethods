all:
	g++ -O3 \
	-I include/eigen-3.4.0 \
	include/metodosIterativos/metodosIterativos.cpp \
	main.cpp -o main && rm results.csv  && ./main
