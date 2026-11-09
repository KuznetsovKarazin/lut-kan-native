#pragma once
#include <stdint.h>
#include <string.h>

// Bounded, newline-framed commands. Oversized/malformed requests never run a test.
struct BenchCommand {
  char text[48];
  uint8_t used=0;
  bool overflow=false;
  bool push(char c) {
    if(c=='\r')return false;
    if(c=='\n') {
      text[used]=0;
      bool valid=!overflow;
      used=0;overflow=false;
      return valid;
    }
    if(used<sizeof(text)-1)text[used++]=c;
    else overflow=true;
    return false;
  }
  bool hello() const {return strcmp(text,"H")==0;}
  const char *token(char verb) const {
    if(strlen(text)!=34 || text[0]!=verb || text[1]!=',')return nullptr;
    for(int i=2;i<34;++i)if(!((text[i]>='0'&&text[i]<='9')||(text[i]>='a'&&text[i]<='f')))return nullptr;
    return text+2;
  }
  const char *session() const {return token('R');}
  const char *probe() const {return token('Q');}
};
