proj4.exe: main.o RiderClass.o CarClass.o AttractionClass.o random.o readInFromFile.o detRiderType.o
	g++ main.o RiderClass.o CarClass.o AttractionClass.o random.o readInFromFile.o detRiderType.o -o proj4.exe

main.o: test_phase3.cpp
	g++ -c test_phase3.cpp -o main.o

RiderClass.o: RiderClass.cpp RiderClass.h
	g++ -c RiderClass.cpp -o RiderClass.o

CarClass.o: CarClass.cpp CarClass.h
	g++ -c CarClass.cpp -o CarClass.o
    
AttractionClass.o: AttractionClass.cpp AttractionClass.h
	g++ -c AttractionClass.cpp -o AttractionClass.o
    
random.o: random.cpp random.h
	g++ -c random.cpp -o random.o
    
readInFromFile.o: readInFromFile.cpp readInFromFile.h
	g++ -c readInFromFile.cpp -o readInFromFile.o
    
detRiderType.o: detRiderType.cpp detRiderType.h
	g++ -c detRiderType.cpp -o detRiderType.o

clean:
	rm -rf main.o RiderClass.o CarClass.o AttractionClass.o random.o readInFromFile.o detRiderType.o proj4.exe
