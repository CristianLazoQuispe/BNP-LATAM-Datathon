3
???`?  ?               @   s?   d dl Z d dlZd dlZd dlZd dlZd dlZd dlZd dl	m
Z
 d dlZd dlZd dlZd dlZd dlmZ e
dd? ?Zdd? Zdd	? Zd
d? Zddd?Zdd? ZdS )?    N)?contextmanager)?InsecureRequestWarningc             c   sT   t j t jt? t j| ? z$y
d V  W n tk
r8   Y nX W d t j t jt j? X d S )N)?signal?SIGALRM?raise_timeout?alarm?TimeoutError?SIG_IGN)?time? r   ?apiquery.py?timeout   s    


r   c             C   s   t ?d S )N)r   )?signum?framer   r   r   r   $   s    r   c             C   s?   d}d}d}t jdtd? xd| r~|dkr~y<tjdd| dd	?}td
|? ?? t|j? t|j? d}W q   |d7 }Y qX qW ytj	|j
?d }|S    td? dt|?kr?td|j
? ?? tt|???Y nX dS )zu
    Add a timeout block, if no answer relaunch the request
    function to query the hackathon api, for scoring
    ?   FN?ignore)?category?
   zdhttps://bnpparibascardif-hackathon.domino-pilot.com:443/models/608a7525d4bd996b9791c91c/latest/model?@G5hH2qmIPdKXUa0CekiVLoW2e9F1eLLwffIhizCyr8Z5Y6K0EPR0AOpV0XCnJaxE)?auth?jsonZverifyzrequests number T?resultz1No result sent by the api. Check your input dict.?textzresponse.text )r   r   )?warnings?filterwarningsr   ?requests?post?printZstatus_code?headersr   ?loadsr   ?dir?	Exception?str)?
input_dict?n?stop?responseZjson_outputr   r   r   ?launch_request(   s2     

r'   c           	   C   s~   t jd?j? } ddd| dddd?i}t|?}td	|? ?? d
dddddddd?}x |D ]}|| || ksVt?qVW td? dS )zFtest if the api launcher return the correct result on a simple examplezrand_y_pred.parquet?data?test_marketingzgautier-ddlza model api test?None?False)?competition_name?	user_name?y_pred?sub_name?holdout_key?
update_ldbz	answer : ?noneg??Bt?I@gоTu?_??g?O??????)?	file_pathr,   ?name?result_csv_file?score?score2?score3r/   z[93m api score okNg??Bt?I?gоTu?_??g?O????߿)?pd?read_parquet?to_jsonr'   r   ?AssertionError)r.   r#   ?answer?correct_answer?keyr   r   r   ?test_launcher_requestO   s*    

r@   r)   ?my_submissionr*   ?Truec             C   sr   t | tj?r| jd?j? } nt | tj?r:tj| dd?j? } d||dkrPtjd?n|| j	? |||d?i}t
|?}|S )a6  
    y_pred      : one pandas series with the prediction and the correct index
                  depending on the competition. If there is a holdout the index
                  should be made of the test index + the holdout index.
                  For instance in the data for marketing competition it was the test client id
                  for the test set and the holdout client id for the holdout.
    competition_name : the name of the competition.
    subname     : submission name, to distinguish your submissions.
    holdout_key : for admin.
    update_ldb  : True by default, False is reserved for admin. Please note
                  that for the leaderboard to be updated the competition
                  conditions must be respected. Opened competition,
                  nb of submissions, time, etc.
    ?pred)r4   r(   r*   ZDOMINO_STARTING_USERNAME)r,   r-   r.   r/   r0   r1   )?
isinstancer9   ZSeries?renameZto_frame?np?ndarray?os?getenvr;   r'   )r.   r,   ?subnamer0   r1   ?usernamer#   r=   r   r   r   ?
submit_apio   s    
rL   c           
   C   sh   t jd?} d}d}d}d}t| ||||?}dddddddddd?	}x |D ]}|| || ksHt?qHW |S )z&
    test the submit_api function
    zrand_y_pred.parquetr)   ztest submit_apir*   r+   r2   zSubmission validated.zgautier-ddlg??Bt?I@gоTu?_??g?O??????)	r3   ?messager,   r4   r5   r6   r7   r8   r/   g??Bt?I?gоTu?_??g?O????߿)r9   r:   rL   r<   )r.   r,   rJ   r0   r1   r=   r>   r?   r   r   r   ?test_submit_api?   s&    


rN   )r)   rA   r*   rB   r*   )?pdb?numpyrF   ?pandasr9   r   rH   ?rer   ?
contextlibr   ?sysr   ?datetimer   Z$requests.packages.urllib3.exceptionsr   r   r   r'   r@   rL   rN   r   r   r   r   ?<module>   s,   '"    
*