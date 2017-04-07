#include <iostream>
using namespace std;
#include "LinkedNodeClass.h"
#include "FIFOQueueClass.h"

int main()
{
  
  int num = 10;
  int outVal(0);
  //int array[num] = {3,2,1,8,1,1,100,1,8,8};
  int array[num] = {1,1,1,1,2,3,8,8,8,100};
  FIFOQueueClass< int > testQueue;
  testQueue.getNumElems();
  
  /*
  testList.getNumElems();
  for (int i=0; i < num; i++)
  {
    testList.insertValue(array[i]);
    testList.getNumElems();
  }
  testList.printForward();
  
  testList.getElemAtIndex(1, outVal);
  testList.getElemAtIndex(num, outVal);
  testList.getElemAtIndex(6, outVal);
  testList.getElemAtIndex(num + 1, outVal);
    
  SortedListClass< int > testList2(testList);
  cout << "Print link after copy constructor!" << endl;
  testList2.printForward();
  testList2.printBackward();
  
  int theVal;
  testList2.removeLast(theVal);
  testList2.printForward();
  
  testList2.removeFront(theVal);
  testList2.printForward();
  
  cout << testList2.getNumElems() << endl;
  
  testList2.clear();
  testList2.printForward();
  */
  
  /*
  cout << "Insert value to testList2!" << endl;
  testList2.insertValue(100);
  testList2.printForward();
  
  cout << "Insert value to testList!" << endl;
  testList.insertValue(1);
  testList.insertValue(99);
  testList.printForward();
    
  cout << "Clear original list!" << endl;
  testList.clear();
  cout << "Print original list after clear!" << endl;
  testList.printForward();
  
  cout << "Print out copied list!" << endl;
  testList2.printForward();
  */
  
    
  
  
}
