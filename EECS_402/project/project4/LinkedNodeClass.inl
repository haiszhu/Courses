#include "LinkedNodeClass.h"

//Only one constructor may be used in this project!
template< class T >
LinkedNodeClass<T>::LinkedNodeClass(
        LinkedNodeClass *inPrev, //Address of node that comes before this one
        const T &inVal,          //Value to be contained in this node
        LinkedNodeClass *inNext  //Address of node that comes after this one
        )
{
  prevNode = inPrev;
  nodeVal = inVal;
  nextNode = inNext;
}

//Returns the value stored within this node.
template< class T >
T LinkedNodeClass<T>::getValue() const
{
  return nodeVal;
}

//Returns the address of the node that follows this node.
template< class T >
LinkedNodeClass<T>* LinkedNodeClass<T>::getNext() const
{
  return nextNode;
}

//Returns the address of the node that comes before this node.
template< class T >
LinkedNodeClass<T>* LinkedNodeClass<T>::getPrev() const
{
  return prevNode;
}

//Sets the object’s next node pointer to NULL.
template< class T >
void LinkedNodeClass<T>::setNextPointerToNull()
{
  nextNode = NULL;
}

//Sets the object's previous node pointer to NULL.
template< class T >
void LinkedNodeClass<T>::setPreviousPointerToNull()
{
  prevNode = NULL;
}

//Sets the previous node of THIS object to point its nextNode to THIS object
//Sets the next node of THIS object to point its prevNode to THIS object
template< class T >
void LinkedNodeClass<T>::setBeforeAndAfterPointers()
{
  (*prevNode).nextNode = this;
  (*nextNode).prevNode = this;
}