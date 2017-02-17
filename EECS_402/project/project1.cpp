//two player guess number game
//01/26/2017 Hai Zhu

#include <iostream>
using namespace std;

//this function get number input within min and max value
int getIntInRange(int minValue, int maxValue)
{
  int  typeInNum;               //input number to be returned
  int  typeInCheck = 0;         //status check weather within range
   
  //loop until keyboard input is within range 
  while (typeInCheck == 0)
  {
    cout << "Enter a number from "<< minValue << " to " << maxValue <<\
" (inclusive): "; //Prompt
    cin >> typeInNum;
        
    if (typeInNum < minValue || typeInNum > maxValue)
    {
      cout << "ERROR: Value is not within valid range. Try again." << endl;
    }
    else
    {
      typeInCheck = 1;
    }
    }
      
  return (typeInNum);
}


//this function get secret number from player 1
int getSecretNumber(int minValue, int maxValue)
{
  const int  clearNum = 30;          //clear screen parameter
  int        secretNum;              //record secret number to be returned
  int        num;                    //iteration variable

  cout << "Player 1: You will enter a secret number for the other player to\
 try to guess" << endl;
  secretNum =  getIntInRange(minValue, maxValue);
  
  //clear screen to prevent next player seeing secret number
  for (num = 1; num <= clearNum; num++)
  {
    cout << "Clearing screen!" << endl;
  }

  return (secretNum);
}


//this function get guess number from player 2 and check
bool getGuessAndCheck(int minValue, int maxValue, int secretNum)
{
  int  guessNum;                    //keyboard input guess number
  
  cout << "Player 2: You will try to guess the secret number" << endl;  
  guessNum = getIntInRange( minValue, maxValue);
 
  //print guess result for player 2
  if (guessNum == secretNum)
  {
    cout << "Congratulations! You guessed it!" << endl;
    return true;
  }
  else if (guessNum > secretNum)
  {
    cout << "The secret number is LOWER than your guess" << endl;
    return false; 
  }
  else
  {
    cout << "The secret number is HIGHER than your guess" << endl;
    return false;
  }

}


//guess game main function
int main(void)
{
  const int  minValue = 1;
  const int  maxValue = 100;
  int        secretNum;                //secret num from player 1
  int        guessCount = 0;           //num of guesses of player 2
  bool       guessResult;              

  //get secret number from player 1
  secretNum = getSecretNumber( minValue, maxValue);
  
  //get and check guess number from player 2
  do
  {
    guessResult = getGuessAndCheck( minValue, maxValue, secretNum);
    guessCount++;
  }
  while (guessResult == false);

  cout << "It took you " << guessCount << " guesses to guess it!" << endl;
  
  return (0);

}
