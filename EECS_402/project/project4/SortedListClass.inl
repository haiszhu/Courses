#include "SortedListClass.h"

//Default Constructor. Will properly initialize a list to
//be an empty list, to which values can be added.
template< class T >
SortedListClass<T>::SortedListClass()
{
  std::cout << "default list ctor was called!" << std::endl;
  head = NULL;
  tail = NULL;
}

//Copy constructor. Will make a complete copy of the list, such
//that one can be changed without affecting the other.
template< class T >
SortedListClass<T>::SortedListClass(const SortedListClass< T > &rhs)
{
  std::cout << "copy ctor was called!" << std::endl;
  head = rhs.head;
  tail = rhs.tail;
}

//Clears the list to an empty state without resulting in any
//memory leaks.
template< class T >
void SortedListClass<T>::clear()
{
  bool status = true;
  while (status)
  {
    if (head == NULL || tail == NULL)
    {
      std::cout << "List is already empty!" << std::endl;
      status = false;
    }
    else
    {
      //LinkedNodeClass *temp;
      // START HERE NEXT
      status = false;
    }
  }
}
