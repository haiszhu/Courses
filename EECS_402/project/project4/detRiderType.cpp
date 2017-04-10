/* ----------------------------------------------------------------------
 * detRiderType.cpp
 *
 * 04/10/2017 Hai Zhu
 *
 * global function to determine rider type, need to be modified
 * if more lines were introduced
 ------------------------------------------------------------------------
 */

#include <iostream>
using namespace std;
#include "constants.h"

int detRiderType(int riderTypeID, int percentSFP, int percentFP)
{
  if (riderTypeID < percentSFP)
  {
    return(RIDER_SFP);
  }
  else if (riderTypeID < percentSFP + percentFP)
  {
    return(RIDER_FP);
  }
  else
  {
    return(RIDER_STD);
  }
}

