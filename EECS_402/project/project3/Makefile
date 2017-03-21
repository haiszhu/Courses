proj3.exe: main.o ColorImageClass.o ColorClass.o PixelLocationClass.o \
           RectangleClass.o
	g++ main.o ColorImageClass.o ColorClass.o PixelLocationClass.o    \
           RectangleClass.o -o proj3.exe

main.o: main.cpp
	g++ -c main.cpp -o main.o

ColorImageClass.o: ColorImageClass.cpp ColorImageClass.h
	g++ -c ColorImageClass.cpp -o ColorImageClass.o

ColorClass.o: ColorClass.cpp ColorClass.h
	g++ -c ColorClass.cpp -o ColorClass.o

PixelLocationClass.o: PixelLocationClass.cpp PixelLocationClass.h
	g++ -c PixelLocationClass.cpp -o PixelLocationClass.o

RectangleClass.o: RectangleClass.cpp RectangleClass.h
	g++ -c RectangleClass.cpp -o RectangleClass.o

clean:
	rm -rf main.o ColorImageClass.o ColorClass.o PixelLocationClass.o \
           RectangleClass.o proj3.exe
