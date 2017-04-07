#include <iostream>
#include "LinkedNodeClass.h"

//Default Constructor. Will properly initialize a list to
//be an empty list, to which values can be added.
template< class T >
SortedListClass<T>::SortedListClass()
{
  cout << "Default ctor was called!" << endl;
  //assign both head and tail to null pointer
  head = NULL;
  tail = NULL;
}

//Copy constructor. Will make a complete copy of the list, such
//that one can be changed without affecting the other.
template< class T >
SortedListClass< T >::SortedListClass(const SortedListClass< T > &rhs)
{
  cout << "Copy ctor was called!" << endl;
  //assign head and tail to null in case &rhs is an empty list
  head = NULL;
  tail = NULL;
  //deep copy of a list
  if (rhs.head != NULL)
  {
    //temp pointer to loop over list
    LinkedNodeClass< T >* temp = rhs.head;
    while (temp != NULL)
    {
      //insert value, and also assign head and tail to correct
      //address when inserting the first node in insertValue fun
      insertValue(temp->getValue());
      temp = temp->getNext();
    }
  }
  else
  {
    cout << "List is empty" << endl;
  }
  //cout << head << endl;

}

//Clears the list to an empty state without resulting in any 
//memory leaks.
template < class T >
void SortedListClass< T >::clear()
{
  //head gets update after one node gets removed from list
  while (head != NULL)
  {
    //dummy variable to utilize removeFront fun
    T temp;
    removeFront(temp);
  }
  cout << "List gets cleaned!" << endl;
}

//Allows the user to insert a value into the list in the 
//appropriate position
template < class T >
void SortedListClass< T >::insertValue(const T &valToInsert)
{
  LinkedNodeClass< T >* newNode;  //newNode in link
  LinkedNodeClass< T >* temp = head;  //point to insertation position
  
  //head is null, empty list
  if (temp == NULL) 
  {
    newNode = new LinkedNodeClass< T >(temp, valToInsert, temp);
    head = newNode;
    tail = newNode;
    cout << "The list is empty before inserting " << valToInsert << endl;
    //cout << head << endl;
  }
  
  //list is not empty
  else  
  {
    //get temp to the right place
    while (temp->getValue() < valToInsert && temp != tail)
    {
      temp = temp->getNext();
    }
    
    //insert at the beginning
    if (temp == head && valToInsert <= temp->getValue()) 
    {
      newNode = new LinkedNodeClass< T >(NULL, valToInsert, temp);
      head = newNode;   //modify head to point to head
      cout << "Insert value " << valToInsert << " at the beginning of a list!" << endl;
    }
    
    //insert at the end
    else if (temp == tail && tail->getValue() < valToInsert) 
    {
      newNode = new LinkedNodeClass< T >(tail, valToInsert, NULL);
      tail = newNode;   //modify tail to point to tail
      cout << "Insert value " << valToInsert << " at the end of a list!" << endl;
    }
    
    //insert in the middle
    else  
    {
      newNode = new LinkedNodeClass< T >(temp->getPrev(),valToInsert, temp);
      cout << "Insert value " << valToInsert << " in the middle of a list!" << endl;
      //cout << head << endl;
      //cout << temp->getPrev() << endl;
    }
    
    //update pointer for adjacent nodes
    newNode->setBeforeAndAfterPointers();
  }
}

//Prints the contents of the list from head to tail to the screen.
template < class T >
void SortedListClass< T >::printForward() const
{
  const LinkedNodeClass< T >* temp = head;
  if (temp != NULL)
  {
    cout << "Forward List Contents Follow:" << endl;
    //loop through list to print all nodes
    while(temp != NULL)
    {
      cout << "  " << temp->getValue(); //<< endl;
      temp = temp->getNext();
    }
    cout << endl;
    cout << "End of List Contents" << endl;
  }
  else
  {
    cout << "The list to be printed is empty!" << endl;
  }
}

//Prints the contents of the list from tail to head to the screen.
template < class T >
void SortedListClass< T >::printBackward() const
{
  const LinkedNodeClass< T >* temp = tail;
  if (temp != NULL)
  {
    cout << "Backward List Contents Follow:" << endl;
    //loop through list to print all nodes
    while (temp != NULL)
    {
      cout << "  " << temp->getValue(); // << endl;
      temp = temp->getPrev();
    }
    cout << endl;
    cout << "End of List Contents" << endl;
  }
  else
  {
    cout << "The list to be printed is empty!" << endl;
  }
}


//Removes the front item from the list and returns the value that 
//was contained in it via the reference parameter.
template < class T >
bool SortedListClass< T >::removeFront(T &theVal)
{
  bool removeStatus = false;
  LinkedNodeClass< T >* pDelNode;
  
  //list is empty
  if (head == NULL)
  {
    cout << "List is empty, removing front fails!" 
         << endl << endl;
  }
  //list is not empty
  else if (head == tail)
  {
    delete head;
    head = NULL;
    tail = NULL;
    removeStatus = true;
    cout << "After removal, this list is empty!" 
         << endl << endl;
  }
  else 
  {
    //get value to be removed
    theVal = head->getValue(); 
    pDelNode = head;
    head = head->getNext(); //correct head accordingly
    //clear memory 
    delete pDelNode;
    //correct pointer of 1st node
    head->setPreviousPointerToNull(); 
    removeStatus = true;
  }
  return (removeStatus);
}

//Removes the last item from the list and returns the value that 
//was contained in it via the reference parameter.
template < class T >
bool SortedListClass< T >::removeLast(T &theVal)
{
  bool removeStatus = false;
  LinkedNodeClass< T >* pDelNode;
  
  //list is empty
  if (tail == NULL)
  {
    cout << "List is empty, removing last fails!" << endl;
  }
  else if (head == tail)
  {
    delete tail;
    head = NULL;
    tail = NULL;
    removeStatus = true;
  }
  else
  {
    //get value to be removed
    theVal = tail->getValue();
    pDelNode = tail;
    tail = tail->getPrev();
    //clear memory
    delete pDelNode;
    //correct pointer of last node
    tail->setNextPointerToNull();
    removeStatus = true;
  }
  return (removeStatus);
}

//Returns the number of nodes contained in the list.
template < class T >
int SortedListClass< T >::getNumElems() const
{
  LinkedNodeClass< T >* temp = head;
  int numElems(0);
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
  cout << "There are " << numElems << " nodes in this list!" << endl;
  cout << endl;
  return (numElems);
}

//Provides the value stored in the node at index provided in the 
//"index" parameter.
template < class T >
bool SortedListClass< T >::getElemAtIndex(const int index, T &outVal)
{
  bool getElemStatus = false;
  LinkedNodeClass< T >* temp = head; 
  
  //if not an empty list
  if (temp != NULL)
  {
    int currentPosition(0);
    //loop through list to print all nodes
    while(temp != tail && currentPosition != index)
    {
      //sync pointer to node with current position
      temp = temp->getNext();
      currentPosition = currentPosition + 1;
    }
    //if index was found
    if (currentPosition == index)
    {
      outVal = temp->getValue();
      getElemStatus = true;
      cout << "The element at index on your list " << index 
           << " is:" << endl;
      cout << "  " << outVal 
           << endl << endl;
    }
    //index greater than number of elements in list
    else
    {
      cout << "You are trying to access index more "
           <<"than total elements in this list!" 
           << endl << endl;
    }
  }
  //empty list
  else
  {
    cout << "You are trying to get node from an empty list!" 
         << endl << endl;
  }
  return (getElemStatus);
}






