import numpy as np

def all_paulis():
    identity = np.array([[1.0 +0.j , 0.0 +0.j], [0.0 +0.j , 1.0 +0.j]])
    sigma_x = np.array([[0.0 +0.j , 1.0 +0.j], [1.0 +0.j , 0.0 +0.j]])
    sigma_y = np.array([[0.0 +0.j , 0.0 - 1.j], [0.0 +1.j , 0.0 +0.j]])
    sigma_z = np.array([[1.0 +0.j , 0.0 +0.j], [0.0 +0.j , -1.0 +0.j]])
    return {'I': identity,'X': sigma_x, 'Y': sigma_y, 'Z': sigma_z}

def sentence_to_operator(word_list):
    paulis = all_paulis()
    operator_list =[]
    for word in word_list:
        operator = 1
        for letter in word:
            operator = np.kron(operator, paulis[letter])
        operator_list.append(operator)
    return operator_list

def get_model(model_name):
    Sx = np.array([[0,1],[1,0]])
    Sy = np.array([[0,-1j],[1j,0]])
    Sz = np.array([[1,0],[0,-1]])
    Sid = np.eye(2)
    if model_name == 'Heisenberg_model':
        H = [(np.kron(np.kron(np.kron(Sx,Sx),Sid),Sid)).astype('complex64'), (np.kron(np.kron(np.kron(Sy,Sy),Sid),Sid)).astype('complex64') , (np.kron(np.kron(np.kron(Sz,Sz),Sid),Sid)).astype('complex64'),
        (np.kron(np.kron(Sid,np.kron(Sx,Sx)),Sid)).astype('complex64'), (np.kron(np.kron(Sid,np.kron(Sy,Sy)),Sid)).astype('complex64'), (np.kron(np.kron(Sid,np.kron(Sz,Sz)),Sid)).astype('complex64'),
        (np.kron(np.kron(Sid,Sid),np.kron(Sx,Sx))).astype('complex64'), (np.kron(np.kron(Sid,Sid),np.kron(Sy,Sy))).astype('complex64'), (np.kron(np.kron(Sid,Sid),np.kron(Sz,Sz))).astype('complex64'),
        (np.kron(np.kron(np.kron(Sx,Sid),Sid),Sx)).astype('complex64'),(np.kron(np.kron(np.kron(Sy,Sid),Sid),Sy)).astype('complex64'), (np.kron(np.kron(np.kron(Sz,Sid),Sid),Sz)).astype('complex64'),(np.kron(np.kron(np.kron(Sz,Sid),Sid),Sid)).astype('complex64'),(np.kron(np.kron(np.kron(Sid,Sz),Sid),Sid)).astype('complex64'), (np.kron(np.kron(np.kron(Sid,Sid),Sz),Sid)).astype('complex64'),(np.kron(np.kron(np.kron(Sid,Sid),Sid),Sz)).astype('complex64')]
        w = [1]*16
    elif model_name == 'H2':
        Hwords = ['ZZ','ZI','IZ','XX']
        w = [0.011280,0.397936,0.397936,0.180931]
        H = sentence_to_operator(Hwords)
    
    elif model_name == 'LiH':
        Hwords = ['ZIII', 'ZZII', 'IZII', 'IIZI', 'IIZZ', 'IIIZ', 'ZIZI', 'ZIZZ','ZIIZ', 'ZZZI', 'ZZZZ', 'ZZIZ', 'IZZI', 'IZZZ', 'IZIZ',
             'XZII', 'XIII', 'IIXZ', 'IIXI', 'XZXZ', 'XZXI', 'XIXZ', 'XIXI', 'XZIZ', 'XIIZ', 'IZXZ', 'IZXI',
             'XXII','IXII', 'IIXX', 'IIIX', 'XIXX','XIIX', 'XXXI', 'XXXX', 'XXIX', 'IXXI', 'IXXX', 'IXIX',
             'YYII','IIYY', 'YYYY',
             'ZXII','IIZX', 'ZIZX', 'ZIIX', 'ZXZI', 'IXZI','ZXZX', 'ZXIX', 'IXZX',
             'ZIXZ', 'ZIXI', 'ZZXZ', 'ZZXI', 
             'ZIXX','ZZXX', 'ZZIX', 'IZXX', 'IZIX', 
             'ZIYY', 'ZZYY', 'IZYY',
             'XZZI','XIZI', 'XZZZ', 'XIZZ',
             'XZXX', 'XZIX', 'XZYY', 'XIYY','XZZX', 
             'XIZX', 'IZZX', 'XXZI', 'XXZZ', 'XXIZ', 'IXZZ', 'IXIZ', 'YYZI', 
             'YYZZ', 'YYIZ', 'XXXZ', 'IXXZ', 'YYXZ', 'YYXI', 'XXYY', 'IXYY', 
             'YYXX', 'YYIX', 'XXZX', 'YYZX', 'ZZZX', 'ZXXZ', 'ZXXI', 'ZXIZ', 
             'ZXXX', 'ZXYY', 'ZXZZ']
        w = [-0.096022, -0.206128, 0.364746, 0.096022, -0.206128, -0.364746, -0.145438,
             0.05604, 0.110811, -0.05604, 0.080334, 0.063673, 0.110811, -0.063673, 
             -0.095216, -0.012585, 0.012585, 0.012585, 0.012585, -0.002667, -0.002667,
             0.002667, 0.002667, 0.007265, -0.007265, 0.007265, 0.007265, -0.02964, 
             0.002792, -0.02964, 0.002792, -0.008195, -0.001271, -0.008195, 0.028926,
             0.007499, -0.001271, 0.007499, 0.009327, 0.02964, 0.02964, 0.028926, 0.002792,
             -0.002792, -0.016781, 0.016781, -0.016781, -0.016781, -0.009327, 0.009327,
             -0.009327, -0.011962, -0.011962, 0.000247, 0.000247, 0.039155, -0.002895, 
             -0.009769, -0.02428, -0.008025, -0.039155, 0.002895, 0.02428, -0.011962, 
             0.011962, -0.000247, 0.000247, 0.008195, 0.001271, -0.008195, 0.008195,
             -0.001271, 0.001271, 0.008025, -0.039155, -0.002895, 0.02428, -0.009769,
             0.008025, 0.039155, 0.002895, -0.02428, -0.008195, -0.001271, 0.008195, 
             0.008195, -0.028926, -0.007499, -0.028926, -0.007499, -0.007499, 0.007499,
             0.009769, -0.001271, -0.001271, 0.008025, 0.007499, -0.007499, -0.009769]
        H = sentence_to_operator(Hwords)
    elif model_name == 'BeH2':
    
        Hwords = ['ZIIIII', 'ZZIIII', 'IZZIII', 'IIZIII', 'IIIZII', 'IIIZZI', 'IIIIZZ',
             'IIIIIZ', 'IZIIII', 'ZZZIII', 'ZIZIII', 'ZIIZII', 'ZIIZZI', 'ZIIIZZ',
             'ZIIIIZ', 'ZZIZII', 'ZZIZZI', 'ZZIIZZ', 'ZZIIIZ', 'IZZZII', 'IZZZZI',
             'IZZIZZ', 'IZZIIZ', 'IIZZII', 'IIZZZI', 'IIZIZZ', 'IIZIIZ', 'IIIIZI',
             'IIIZZZ', 'IIIZIZ', 'XZIIII', 'XIIIII', 'IZXIII', 'IIXIII', 'IIIXZI', 
             'IIIXII', 'IIIIZX', 'IIIIIX', 'XIXIII', 'XZXIII', 'XZIXZI', 'XZIXII',
             'XIIXZI', 'XIIXII', 'XZIIZX', 'XZIIIX', 'XIIIZX', 'XIIIIX', 'IZXXZI',
             'IZXXII', 'IIXXZI', 'IIXXII', 'IZXIZX', 'IZXIIX', 'IIXIZX', 'IIXIIX', 
             'IIIXIX', 'IIIXZX', 'ZZXIII', 'ZIXIII', 'ZIIXZI', 'ZIIXII', 'ZIIIZX', 
             'ZIIIIX', 'ZZIXZI', 'ZZIXII', 'ZZIIZX', 'ZZIIIX', 'XIZIII', 'XZZIII', 
             'XZIZII', 'XIIZII', 'XZIZZI', 'XIIZZI', 'XZIIZZ', 'XIIIZZ', 'XZIIIZ', 
             'XIIIIZ', 'YIYIII', 'YYIXXZ', 'YYIIXI', 'IYYXXZ', 'IYYIXI', 'IIIXIZ', 
             'XXZXXZ', 'XXZIXI', 'IXIXXZ', 'IXIIXI', 'IIZXII', 'XXZYYI', 'XXZIYY', 
             'IXIYYI', 'IXIIYY', 'IIIYIY', 'YYIYYI', 'YYIIYY', 'IYYYYI', 'IYYIYY',
             'XXZXXX', 'IXIXXX', 'IIZIIX', 'XXZYXY', 'IXIYXY', 'YYIXXX', 'IYYXXX', 
             'YYIYXY', 'IYYYXY', 'XXZZXZ', 'IXIZXZ', 'YYIZXZ', 'IYYZXZ', 'XXZZXX', 
             'IXIZXX', 'IIIZIX', 'YYIZXX', 'IYYZXX', 'XXXXXZ', 'XXXIXI', 'IIXIIZ', 
             'XXXYYI', 'XXXIYY', 'YXYXXZ', 'YXYIXI', 'YXYYYI', 'YXYIYY', 'XXXXXX', 
             'XXXYXY', 'YXYXXX', 'YXYYXY', 'XXXZXZ', 'IIXZII', 'YXYZXZ', 'XXXZXX', 
             'YXYZXX', 'ZXZXXZ', 'ZXZIXI', 'ZXZYYI', 'ZXZIYY', 'ZXZXXX', 'ZXZYXY', 
             'ZXZZXZ', 'ZXZZXX', 'ZXXXXZ', 'ZXXIXI', 'ZXXYYI', 'ZXXIYY', 'ZXXXXX', 
             'ZXXYXY', 'ZXXZXZ', 'ZXXZXX', 'IZZXZI', 'IZZXII', 'IZZIZX', 'IZZIIX', 
             'IIZXZI', 'IIZIZX', 'IZXZII', 'IZXZZI', 'IIXZZI', 'IZXIZZ', 'IIXIZZ', 
             'IZXIIZ', 'IIIZZX', 'IIIXZZ']
        
        w= [-0.143021, 0.104962, 0.038195, -0.325651, -0.143021, 0.104962, 0.038195, 
            -0.325651, 0.172191, 0.174763, 0.136055, 0.116134, 0.094064, 0.099152, 
            0.123367, 0.094064, 0.098003, 0.102525, 0.097795, 0.099152, 0.102525, 
            0.112045, 0.105708, 0.123367, 0.097795, 0.105708, 0.133557, 0.172191,
            0.174763, 0.136055, 0.05911, -0.05911, 0.161019, -0.161019, 0.05911, -0.05911,
            0.161019, -0.161019, -0.038098, -0.0033, 0.013745, -0.013745, -0.013745, 
            0.013745, 0.011986, -0.011986, -0.011986, 0.011986, 0.011986, -0.011986, 
            -0.011986, 0.011986, 0.013836, -0.013836, -0.013836, 0.013836, -0.038098, 
            -0.0033, -0.002246, 0.002246, 0.014815, -0.014815, 0.009922, -0.009922, -0.002038,
            0.002038, -0.007016, 0.007016, -0.006154, 0.006154, 0.014815, -0.014815, -0.002038,
            0.002038, 0.001124, -0.001124, 0.017678, -0.017678, -0.041398, 0.011583, -0.011094,
            0.010336, -0.005725, -0.006154, 0.011583, -0.011094, -0.011094, 0.026631, -0.017678,
            0.011583, 0.010336, -0.011094, -0.005725, -0.041398, 0.011583, 0.010336, 0.010336, 0.0106,
            0.024909, -0.031035, -0.010064, 0.024909, -0.031035, 0.024909, 0.021494, 0.024909, 0.021494, 
            0.011094, -0.026631, 0.011094, 0.005725, 0.010336, -0.005725, 0.002246, 0.010336,
            0.0106, 0.024909, -0.031035, -0.010064, 0.024909, 0.021494, 0.024909, -0.031035,
            0.024909, 0.021494, 0.063207, 0.063207, 0.063207, 0.063207, 0.031035, -0.009922,
            0.031035, 0.021494, 0.021494, 0.011094, -0.026631, 0.011094, 0.005725, 0.031035,
            0.031035, 0.026631, 0.005725, 0.010336, -0.005725, 0.010336, 0.0106, 0.021494,
            0.021494, 0.005725, 0.0106, 0.001124, -0.001124, -0.007952, 0.007952, 0.017678,
            0.010064, 0.009922, -0.007016, 0.007016, -0.007952, 0.007952, 0.010064, -0.002246, 0.006154]
        H = sentence_to_operator(Hwords)
        
    return (Hwords,w)

def get_hamil(model):
    Hwords, w = get_model(model)
    H = sentence_to_operator(Hwords)
    hamiltonian = np.zeros((len(H[0]), len(H[0])))
    for i in range(len(H)):
        hamiltonian = hamiltonian + H[i]*w[i]
    return hamiltonian

def nth_state(hamiltonian, n):
    evalues, evectors = np.linalg.eig(hamiltonian)
    evectors = np.transpose(evectors)
    ground_index = np.argmin(evalues)
    for i in range(n):
        evalues = np.delete(evalues, [ground_index])
        evectors = np.delete(evectors, ground_index, 0)
        ground_index = np.argmin(evalues)
    return evalues[ground_index], evectors[ground_index]


