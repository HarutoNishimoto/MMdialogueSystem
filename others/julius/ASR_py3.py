# coding:utf-8
import subprocess
import socket
import time
import re
import threading
import sys

class userUtteranceInfo:
    def __init__(self):
        self.us_time = None
        self.word = None

    def addWord(self, msg):
        if self.word != None:
            self.word += msg
        else:
            self.word = msg

    def setWord(self, msg):
        self.word = msg
        
    def setTime(self, time):
        self.us_time = time


######## C:\Users\kaldi\dictation-kit-v4.4\ ここにおいて、ここに移動して、実行してください。
####参考URL => https://teratail.com/questions/108431

def main():
    p = subprocess.Popen("run-win-dnn-module", shell=True)
    print('#### Initiating start ####')
    time.sleep(10)
    print('#### Initiating Done ####')

    host = 'localhost'
    port = 10500
    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((host, port))

    # init
    msg_stock = ''

    # 新規作成
    f = open('userUtteranceInfo.csv', 'w', encoding='utf-8')
    f.write('u_s,word\n')
    f.close()

    while True:
        try:
            message = client.recv(1024).decode("utf-8")

            print(message)

            if len(message) > 1:

                #### 新バージョン
                msg_stock, ts = XMLparser(msg_stock, message)
                if ts != None:
                    print('word', ts.word)
                    print('time', ts.us_time)

                    # 書き出し（追記）
                    f = open('userUtteranceInfo.csv', 'a', encoding='utf-8')
                    print(type(ts.word))
                    f.write('{},{}\n'.format(str(ts.us_time), ts.word))
                    f.close()
                print("---------------------------------------")

        except KeyboardInterrupt:
            print ("KeyboardInterrupt occured.")
            client.close()
            p.kill()
            quit()



def XMLparser(msg_stock, msg):

    msg_stock += msg
    msg_list = msg_stock.split('</RECOGOUT>')

    if len(msg_list) >= 2:
        ts = userUtteranceInfo()

        xmlString = msg_list[0].split('\n')
        xmlString = [x for x in xmlString if '=' in x]

        dicList = []
        for line in xmlString:
            dic = makeTSdic(line)
            dicList.append(dic)

        for d in dicList:
            if 'STARTREC' in d.values() and 'TIME' in d.keys():
                ts.setTime(int(d['TIME']))
            if 'WORD' in d.keys():
                ts.addWord(d['WORD'])

        msg_stock = ''.join(msg_list[1:])
        return msg_stock, ts
    else:
        msg_stock = ''.join(msg_list[:])
        return msg_stock, None

# str -> dict
def makeTSdic(xmlstr):
    retstr = xmlstr.split(' ')
    retstr = [x for x in retstr if x != '']
    retstr = ' '.join(retstr)
    retstr = retstr.replace('<>', '')
    retstr = retstr.translate(str.maketrans({'<': None, '"': None, '>': None, '/': None}))
    retstr = re.split('[ =]', retstr)

    Dict = {}
    for i, val in enumerate(retstr):
        if val == 'STATUS':
            Dict['STATUS'] = retstr[i+1]
        elif val == 'TIME':
            Dict['TIME'] = retstr[i+1]
        elif val == 'WORD':
            Dict['WORD'] = retstr[i+1]

    return Dict


if __name__ == "__main__":

    main()

