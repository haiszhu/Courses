/* ----------------------------------------------------------------------
 * LinkedNodeClass.inl
 *
 * 04/10/2017 Hai Zhu
 *
 * linked node class ctor, dtor, and member function
 ------------------------------------------------------------------------
 */

#include <iostream>

//Only one constructor may be used in this project!
template< class T >
LinkedNodeClass<T>::LinkedNodeClass(
        LinkedNodeClass *inPrev,
        const T &inVal,
        LinkedNodeClass *inNext
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
  return (nodeVal);
}

//Returns the address of the node that follows this node.
template< class T >
LinkedNodeClass<T>* LinkedNodeClass<T>::getNext() const
{
  return (nextNode);
}

//Returns the address of the node that comes before this node.
template< class T >
LinkedNodeClass<T>* LinkedNodeClass<T>::getPrev() const
{
  return (prevNode);
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

//Sets the previous node of THIS object to point its nextNode to
//THIS object Sets the next node of THIS object to point
//its prevNode to THIS object
template< class T >
void LinkedNodeClass<T>::setBeforeAndAfterPointers()
{
  
  if (this->prevNode != NULL)
  {
    this->prevNode->nextNode = this;
  }
  else
  {
    //cout << "Before is null, you are at the beginning of a list!"
    //     << endl;
  }
  //check if this->prevNode is null or not
  if (this->nextNode != NULL)
  {
    this->nextNode->prevNode = this;
  }
  else
  {
    //cout << "After is null, you are at the end of a list!"
    //     << endl;
  }
}
