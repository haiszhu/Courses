/* ----------------------------------------------------------------------
 * printMainMenu.h
 *
 * 03/21/2017 Hai Zhu
 *
 * print main menu function header file
 ------------------------------------------------------------------------
 */
#ifndef _PRINTMAINMENU_H_
#define _PRINTMAINMENU_H_

void printMainMenu()
{
  cout << ANN_REC << ". Annotate image with rectangle" << endl;
  cout << ANN_PAT << ". Annotate image with pattern from file" << endl;
  cout << INS_IMA << ". Insert another image" << endl;
  cout << WRI_OUT << ". Write out current image" << endl;
  cout << EXIT << ". Exit the program" << endl;
  cout << "Enter int for main menu choice: ";
}

#endif