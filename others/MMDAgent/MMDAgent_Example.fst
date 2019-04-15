# ----------------------------------------------------------------- #
#           MMDAgent "Sample Script"                                #
#           released by MMDAgent Project Team                       #
#           http://www.mmdagent.jp/                                 #
# ----------------------------------------------------------------- #
#                                                                   #
#  Copyright (c) 2009-2013  Nagoya Institute of Technology          #
#                           Department of Computer Science          #
#                                                                   #
# Some rights reserved.                                             #
#                                                                   #
# This work is licensed under the Creative Commons Attribution 3.0  #
# license.                                                          #
#                                                                   #
# You are free:                                                     #
#  * to Share - to copy, distribute and transmit the work           #
#  * to Remix - to adapt the work                                   #
# Under the following conditions:                                   #
#  * Attribution - You must attribute the work in the manner        #
#    specified by the author or licensor (but not in any way that   #
#    suggests that they endorse you or your use of the work).       #
# With the understanding that:                                      #
#  * Waiver - Any of the above conditions can be waived if you get  #
#    permission from the copyright holder.                          #
#  * Public Domain - Where the work or any of its elements is in    #
#    the public domain under applicable law, that status is in no   #
#    way affected by the license.                                   #
#  * Other Rights - In no way are any of the following rights       #
#    affected by the license:                                       #
#     - Your fair dealing or fair use rights, or other applicable   #
#       copyright exceptions and limitations;                       #
#     - The author's moral rights;                                  #
#     - Rights other persons may have either in the work itself or  #
#       in how the work is used, such as publicity or privacy       #
#       rights.                                                     #
#  * Notice - For any reuse or distribution, you must make clear to #
#    others the license terms of this work. The best way to do this #
#    is with a link to this web page.                               #
#                                                                   #
# See http://creativecommons.org/ for details.                      #
# ----------------------------------------------------------------- #

# 1st field: state before transition
# 2nd field: state after transition
# 3rd field: event (input message)
# 4th field: command (output message)
#
# Model
# MODEL_ADD|(model alias)|(model file name)|(x position),(y position),(z position)|(x rotation),(y rotation),(z rotation)|(ON or OFF for cartoon)|(parent model alias)|(parent bone name)
# MODEL_CHANGE|(model alias)|(model file name)
# MODEL_DELETE|(model alias)
# MODEL_EVENT_ADD|(model alias)
# MODEL_EVENT_CHANGE|(model alias)
# MODEL_EVENT_DELETE|(model alias)
#
# Motion
# MOTION_ADD|(model alias)|(motion alias)|(motion file name)|(FULL or PART)|(ONCE or LOOP)|(ON or OFF for smooth)|(ON or OFF for repos)
# MOTION_ACCELERATE|(model alias)|(motion alias)|(speed)|(duration)|(specified time for end)
# MOTION_CHANGE|(model alias)|(motion alias)|(motion file name)
# MOTION_DELETE|(mpdel alias)|(motion alias)
# MOTION_EVENT_ADD|(model alias)|(motion alias)
# MOTION_EVENT_ACCELERATE|(model alias)|(motion alias)
# MOTION_EVENT_CHANGE|(model alias)|(motion alias)
# MOTION_EVENT_DELETE|(model alias)|(motion alias)
#
# Move and Rotate
# MOVE_START|(model alias)|(x position),(y position),(z position)|(GLOBAL or LOCAL position)|(move speed)
# MOVE_STOP|(model alias)
# MOVE_EVENT_START|(model alias)
# MOVE_EVENT_STOP|(model alias)
# TURN_START|(model alias)|(x position),(y position),(z position)|(GLOBAL or LOCAL position)|(rotation speed)
# TURN_STOP|(model alias)
# TURN_EVENT_START|(model alias)
# TURN_EVENT_STOP|(model alias)
# ROTATE_START|(model alias)|(x rotation),(y rotaion),(z rotation)|(GLOBAL or LOCAL rotation)|(rotation speed)
# ROTATE_STOP|(model alias)
# ROTATE_EVENT_START|(model alias)
# ROTATE_EVENT_STOP|(model alias)
#
# Sound
# SOUND_START|(sound alias)|(sound file name)
# SOUND_STOP|(sound alias)
# SOUND_EVENT_START|(sound alias)
# SOUND_EVENT_STOP|(sound alias)
#
# Stage
# STAGE|(stage file name)
# STAGE|(bitmap file name for floor),(bitmap file name for background)
#
# Light
# LIGHTCOLOR|(red),(green),(blue)
# LIGHTDIRECTION|(x position),(y position),(z position)
#
# Camera
# CAMERA|(x position),(y position),(z position)|(x rotation),(y rotation),(z rotation)|(distance)|(fovy)|(time)
# CAMERA|(motion file name)
#
# Speech recognition
# RECOG_EVENT_START
# RECOG_EVENT_STOP|(word sequence)
# RECOG_MODIFY|GAIN|(ratio)
# RECOG_MODIFY|USERDICT_SET|(dictionary file name)
# RECOG_MODIFY|USERDICT_UNSET
#
# Speech synthesis
# SYNTH_START|(model alias)|(voice alias)|(synthesized text)
# SYNTH_STOP|(model alias)
# SYNTH_EVENT_START|(model alias)
# SYNTH_EVENT_STOP|(model alias)
# LIPSYNC_START|(model alias)|(phoneme and millisecond pair sequence)
# LIPSYNC_STOP|(model alias)
# LIPSYNC_EVENT_START|(model alias)
# LIPSYNC_EVENT_STOP|(model alias)
#
# Variable
# VALUE_SET|(variable alias)|(value)
# VALUE_SET|(variable alias)|(minimum value for random)|(maximum value for random)
# VALUE_UNSET|(variable alias)
# VALUE_EVAL|(variable alias)|(EQ or NE or LE or LT or GE or GT for evaluation)|(value)
# VALUE_EVENT_SET|(variable alias)
# VALUE_EVENT_UNSET|(variable alias)
# VALUE_EVENT_EVAL|(variable alias)|(EQ or NE or LE or LT or GE or GT for evaluation)|(value)|(TRUE or FALSE)
# TIMER_START|(count down alias)|(value)
# TIMER_STOP|(count down alias)
# TIMER_EVENT_START|(count down alias)
# TIMER_EVENT_STOP|(count down alias)
#
# Plugin
# PLUGIN_ENABLE|(plugin name)
# PLUGIN_DISABLE|(plugin name)
# PLUGIN_EVENT_ENABLE|(plugin name)
# PLUGIN_EVENT_DISABLE|(plugin name)
#
# Other events
# DRAGANDDROP|(file name)
# KEY|(key name)
#
# Other commands
# EXECUTE|(file name)
# KEY_POST|(window class name)|(key name)|(ON or OFF for shift-key)|(ON or OFF for ctrl-key)|(On or OFF for alt-key)

# 0011-0020 Initialization

0    11   <eps>                               MODEL_ADD|bootscreen|Accessory\bootscreen\bootscreen.pmd|0.0,12.85,17.6|0.0,0.0,0.0|OFF
11   12   MODEL_EVENT_ADD|bootscreen          CAMERA|0,18,0|0,0,0|50.0|8|0
12   14   <eps>                               MODEL_ADD|mei|Model\mei\mei.pmd|0.0,0.0,-14.0
14   15   <eps>                               STAGE|Stage\building2\floor.bmp,Stage\building2\background.bmp
15   16   <eps>                               MOTION_ADD|mei|base|Motion\mei_wait\mei_wait.vmd|FULL|LOOP|ON|OFF
16   17   <eps>                               TIMER_START|bootscreen|1.5
17   2    TIMER_EVENT_STOP|bootscreen         MODEL_DELETE|bootscreen

# 0021-0030 Idle behavior

2    21   <eps>                               TIMER_START|idle1|20
21   22   TIMER_EVENT_START|idle1             TIMER_START|idle2|40
22   23   TIMER_EVENT_START|idle2             TIMER_START|idle3|60
23   1    TIMER_EVENT_START|idle3             VALUE_SET|random|0|100
1    1    RECOG_EVENT_START                   MOTION_ADD|mei|listen|Expression\mei_listen\mei_listen.vmd|PART|ONCE
1    1    TIMER_EVENT_STOP|idle1              MOTION_ADD|mei|idle|Motion\mei_idle\mei_idle_boredom.vmd|PART|ONCE
1    1    TIMER_EVENT_STOP|idle2              MOTION_ADD|mei|idle|Motion\mei_idle\mei_idle_touch_clothes.vmd|PART|ONCE
1    2    TIMER_EVENT_STOP|idle3              MOTION_ADD|mei|idle|Motion\mei_idle\mei_idle_think.vmd|PART|ONCE



# 006X Theme sports
1   61  RECOG_EVENT_STOP|sports|hold|1               SYNTH_START|mei|mei_voice_normal|これからスポーツの話をしましょう|op
1   61  RECOG_EVENT_STOP|sports|hold|2              SYNTH_START|mei|mei_voice_normal|さいきんスポーツはしましたか？|qw
1   61  RECOG_EVENT_STOP|sports|hold|3              SYNTH_START|mei|mei_voice_normal|どのスポーツが好きですか？|qw
1   61  RECOG_EVENT_STOP|sports|hold|4              SYNTH_START|mei|mei_voice_normal|学生時代は運動をしていましたか？|qw
1   61  RECOG_EVENT_STOP|sports|hold|5              SYNTH_START|mei|mei_voice_normal|そのスポーツは見るかするならどちらが好きですか？|qw
1   63  RECOG_EVENT_STOP|sports|hold|6             <eps>
1   61  RECOG_EVENT_STOP|sports|dig|1                SYNTH_START|mei|mei_voice_normal|日本では56年ぶりの開催だそうです。|io
1   61  RECOG_EVENT_STOP|sports|dig|2                SYNTH_START|mei|mei_voice_normal|空手が新しい競技として追加されます。|io
1   61  RECOG_EVENT_STOP|sports|dig|3                SYNTH_START|mei|mei_voice_normal|オリンピックは見に行く予定はありますか？|io
1   61  RECOG_EVENT_STOP|sports|dig|4                SYNTH_START|mei|mei_voice_normal|テレビ観戦はするつもりですか？|io
1   63  RECOG_EVENT_STOP|sports|dig|5                <eps>
1   63  RECOG_EVENT_STOP|sports|change|1             <eps>

61  62  <eps>                                           MOTION_ADD|mei|expression|Expression\mei_happiness\mei_happiness.vmd|PART|ONCE
62  2   SYNTH_EVENT_STOP|mei                            <eps>
63  70  <eps>                                           THEME_CHANGE|trip


# 007X Theme trip
70  71  <eps>                                        SYNTH_START|mei|mei_voice_normal|これから旅行の話をしましょう|op
1   71  RECOG_EVENT_STOP|trip|hold|1                 SYNTH_START|mei|mei_voice_normal|旅行は好きですか？|op
1   71  RECOG_EVENT_STOP|trip|hold|2                 SYNTH_START|mei|mei_voice_normal|さいきん旅行に行きましたか？|qy
1   71  RECOG_EVENT_STOP|trip|hold|3                 SYNTH_START|mei|mei_voice_normal|どこに行きましたか？|qy
1   71  RECOG_EVENT_STOP|trip|hold|4                 SYNTH_START|mei|mei_voice_normal|その旅行では何が思い出に残っていますか？|qy
1   73  RECOG_EVENT_STOP|trip|hold|5                 <eps>
1   71  RECOG_EVENT_STOP|trip|dig|1                  SYNTH_START|mei|mei_voice_normal|行ってよかった旅先のランキングが発表されたそうです。|io
1   71  RECOG_EVENT_STOP|trip|dig|2                  SYNTH_START|mei|mei_voice_normal|3位は清水寺、2位は金閣寺、1位は函館のようです。|io
1   71  RECOG_EVENT_STOP|trip|dig|3                  SYNTH_START|mei|mei_voice_normal|京都はいろいろな観光スポットがあるので行ってみたいですね|io
1   71  RECOG_EVENT_STOP|trip|dig|4                  SYNTH_START|mei|mei_voice_normal|京都のほかの観光スポットは行ったことがありますか？|io
1   71  RECOG_EVENT_STOP|trip|dig|5                  SYNTH_START|mei|mei_voice_normal|いつごろの季節に行かれたんですか？|io
1   73  RECOG_EVENT_STOP|trip|dig|6                  <eps>
1   73  RECOG_EVENT_STOP|trip|change|1               <eps>

71  72  <eps>                                           MOTION_ADD|mei|expression|Expression\mei_happiness\mei_happiness.vmd|PART|ONCE
72  2   SYNTH_EVENT_STOP|mei                            <eps>
73  80  <eps>                                           THEME_CHANGE|food


# 008X Theme food
80  81  <eps>                                        SYNTH_START|mei|mei_voice_normal|これから食べ物の話をしましょう|op
1   81  RECOG_EVENT_STOP|food|hold|1                 SYNTH_START|mei|mei_voice_normal|朝ごはんには何をたべていますか？|op
1   81  RECOG_EVENT_STOP|food|hold|2                 SYNTH_START|mei|mei_voice_normal|それはよくたべるんですか？|qy
1   81  RECOG_EVENT_STOP|food|hold|3                 SYNTH_START|mei|mei_voice_normal|好きな食べ物は何ですか？|qy
1   81  RECOG_EVENT_STOP|food|hold|4                 SYNTH_START|mei|mei_voice_normal|それのどういったところが好きなんですか？|qy
1   81  RECOG_EVENT_STOP|food|hold|5                 SYNTH_START|mei|mei_voice_normal|それのどういったところが好きなんですか？|qy
1   83  RECOG_EVENT_STOP|food|hold|6                 <eps>
1   81  RECOG_EVENT_STOP|food|dig|1                  SYNTH_START|mei|mei_voice_normal|ごはんのおとものランキングが発表されたそうです。|io
1   81  RECOG_EVENT_STOP|food|dig|2                  SYNTH_START|mei|mei_voice_normal|3位はうめぼし、2位は昆布、1位は海苔のようです。|io
1   81  RECOG_EVENT_STOP|food|dig|3                  SYNTH_START|mei|mei_voice_normal|コンビニなどのおにぎりの具だとランキングが変わってきそうですね。|io
1   81  RECOG_EVENT_STOP|food|dig|4                  SYNTH_START|mei|mei_voice_normal|おにぎりの具だと何が好きですか？|io
1   83  RECOG_EVENT_STOP|food|dig|5                  <eps>
1   83  RECOG_EVENT_STOP|food|change|1               <eps>

81  82  <eps>                                           MOTION_ADD|mei|expression|Expression\mei_happiness\mei_happiness.vmd|PART|ONCE
82  2   SYNTH_EVENT_STOP|mei                            <eps>
83  90  <eps>                                           THEME_CHANGE|music


# 009X Theme music
90  91  <eps>                                        SYNTH_START|mei|mei_voice_normal|これから音楽の話をしましょう|op
1   91  RECOG_EVENT_STOP|food|hold|1                 SYNTH_START|mei|mei_voice_normal|音楽は聞かれますか？|op
1   91  RECOG_EVENT_STOP|food|hold|2                 SYNTH_START|mei|mei_voice_normal|聞くとしたら洋楽か邦楽のどちらですか？|qy
1   91  RECOG_EVENT_STOP|food|hold|3                 SYNTH_START|mei|mei_voice_normal|好きなアーティストっていますか？|qy
1   91  RECOG_EVENT_STOP|food|hold|4                 SYNTH_START|mei|mei_voice_normal|そのアーティストのどういったところが好きなんですか？|qy
1   91  RECOG_EVENT_STOP|food|hold|5                SYNTH_START|mei|mei_voice_normal|それのどういったところが好きなんですか？|qy
1   93  RECOG_EVENT_STOP|food|hold|6                 <eps>
1   91  RECOG_EVENT_STOP|food|dig|1                  SYNTH_START|mei|mei_voice_normal|年末の紅白歌合戦は見ますか？|io
1   91  RECOG_EVENT_STOP|food|dig|2                  SYNTH_START|mei|mei_voice_normal|最近では？？？が流行してますよね。|io
1   91  RECOG_EVENT_STOP|food|dig|3                  SYNTH_START|mei|mei_voice_normal|結構どこでも流れているイメージですけど、聞いたことありますか？|io
1   93  RECOG_EVENT_STOP|food|dig|4                  <eps>
1   93  RECOG_EVENT_STOP|food|change|1               <eps>

91  92  <eps>                                           MOTION_ADD|mei|expression|Expression\mei_happiness\mei_happiness.vmd|PART|ONCE
92  2   SYNTH_EVENT_STOP|mei                            <eps>
93  100  <eps>                                           THEME_CHANGE|music



# 009X 終了のあいさつ

100  2  <eps>                                           SYNTH_START|mei|mei_voice_normal|これで対話は終了です。ありがとうございました。|op
