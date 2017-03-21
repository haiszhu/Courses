/* ----------------------------------------------------------------------
 * inOptionStatusCheck.h
 *
 * 03/21/2017 Hai Zhu
 *
 * check status of read in user option
 ------------------------------------------------------------------------
 */
#ifndef _INOPTIONSTATUSCHECK_H_
#define _INOPTIONSTATUSCHECK_H_

#include "constants.h"

void inOptionStatusCheck(int &usrOption)
{
  bool validInputFound = false;
  while (!validInputFound)
  {
    cin >> usrOption;
    if (cin.fail()) //if in fail state
    {
      cin.clear();
      cin.ignore(SKIP, '\n');
      cout << "Try again with integer value please! Enter";
    }
    else
    {
      validInputFound = true;
    }
  }
}

#endif
