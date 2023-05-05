import sys
NUMBER=1
log_file = open(sys.argv[1])
if len(sys.argv) == 3 and sys.argv[2] == "mnli":
    MNLI=True
else:
    MNLI=False
log_file = open(sys.argv[1])
lines = log_file.readlines()
acc_list = []
acc = []
for line in lines:
    if ">>   eval_accu" in line:
        score = float(line.strip()[-6:]) * 100
        acc.append(str(round(score, 2)))
    if (len(acc) == NUMBER*2 and MNLI) or (len(acc) == NUMBER and not MNLI):
        acc_list.append(acc)
        acc = []
if MNLI:
    for i in range(NUMBER):
        print("\t".join([x[2*i] for x in acc_list]))
    for i in range(NUMBER):
        print("\t".join([x[2*i+1] for x in acc_list]))
else:
    for i in range(NUMBER):
        print("\t".join([x[i] for x in acc_list]))
