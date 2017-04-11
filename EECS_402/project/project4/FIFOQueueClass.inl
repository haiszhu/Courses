/* -------------------------------------------------------------------
 * FIFOQueueClass.inl
 *
 * 04/10/2017 Hai Zhu
 *
 * first in first out queue class ctor, dtor, and member function
 *
 * (notes: to actually use print() function, the template accept int
 * and char directly, or your class T needs to have getValue() member
 * function to access value you want to get printed, and based on
 * that this memeber function needs to be modified accordingly)
 ---------------------------------------------------------------------
 */

#include <iostream>
#include "LinkedNodeClass.h"
#include "constants.h"

//Default Constructor. Will properly initialize a queue to 
//be an empty queue, to which values can be added. 
template < class T >
FIFOQueueClass< T >::FIFOQueueClass()
{
  head = NULL;
  tail = NULL;
}

//Inserts the value provided (newItem) into the queue. 
template < class T >
void FIFOQueueClass< T >::enqueue(const T &newItem)
{
  LinkedNodeClass< T >* newNode;  //newNode in link
  
  //head is null, empty list
  if (head == NULL && tail == NULL) 
  {
    newNode = new LinkedNodeClass< T >(head, newItem, tail);
    head = newNode;
    tail = newNode;
    //cout << "The list is empty before inserting "
    //     << newItem << endl;
    //cout << head << endl;
  }
  //list is not empty
  else if (head != NULL && tail != NULL)
  { 
    newNode = new LinkedNodeClass< T >(tail, newItem, NULL);
    tail = newNode;   //modify tail to point to tail
    //cout << "Insert value " << newItem
    //     << " at the end of a list!"
    //     << endl;
    //update pointer for adjacent nodes
    newNode->setBeforeAndAfterPointers();
  } 
  //head and tail not consistent
  else
  {
    //cout << "Only one of head or tail is pointing to NULL!"
    //     << endl;
  }
  
}

//Attempts to take the next item out of the queue.
template < class T >
bool FIFOQueueClass< T >::dequeue(T &outItem)
{
  bool dequeueStatus = false;
  LinkedNodeClass< T >* pDelNode;
  
  //list is empty
  if (head == NULL)
  {
    //cout << "List is empty, removing front fails!" 
    //     << endl << endl;
  }
  //list is not empty
  else if (head == tail)
  {
    outItem = head->getValue(); 
    
    delete head;
    head = NULL;
    tail = NULL;
    dequeueStatus = true;
    //cout << "After removal, this queue is empty!" 
    //     << endl << endl;
  }
  else 
  {
    //get value to be removed
    outItem = head->getValue(); 
    pDelNode = head;
    head = head->getNext(); //correct head accordingly
    //clear memory 
    delete pDelNode;
    //correct pointer of 1st node
    head->setPreviousPointerToNull(); 
    dequeueStatus = true;
  }
  return (dequeueStatus);
}


//Prints out the contents of the queue.
template < class T >
void FIFOQueueClass< T >::print() const
{
  const LinkedNodeClass< T >* temp = head;
  if (temp != NULL)
  {
    cout << "Forward Queue Contents Follow:" << endl;
    //loop through list to print all nodes
    while(temp != NULL)
    {
      
      T tempValue;
      tempValue = temp->getValue();
      int info;
      cout << " " << temp ;
      temp = temp->getNext();
    }
    cout << endl;
    cout << "End of Queue Contents" << endl;
  }
  else
  {
    cout << "The queue to be printed is empty!" << endl;
  }
}

//Returns the number of nodes contained in the list.
template < class T >
int FIFOQueueClass< T >::getNumElems() const
{
  LinkedNodeClass< T >* temp = head;
  int numElems(INI_VALUE);
  if (temp != NULL)
  {
    //we are at 1st node
    numElems = numElems + 1;
    while (temp != tail)
    {
      //move to next node as long as it is not tail yet.
      temp = temp->getNext();
      numElems = numElems + 1;
    } //get to tail position, count is over
  }
  //cout << "There are " << numElems
  //     << " nodes in this queue!" << endl;
  //cout << endl;
  return (numElems);
}
