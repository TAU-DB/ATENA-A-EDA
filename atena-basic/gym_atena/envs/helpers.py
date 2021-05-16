import numpy as np
import operator
from copy import deepcopy
import scipy
import scipy.stats
from scipy.stats import entropy


def get_dict(dict_str):
    if type(dict_str) is not str:
        return dict_str
    else:
        try:
            return ast.literal_eval(dict_str)
        except:
            print(dict_str)
            return {}


def hack_min(pd_series):
    return np.min(pd_series.dropna())


def hack_max(pd_series):
    return np.max(pd_series.dropna())


OPERATOR_TYPE_LOOKUP = {
        0: "back",
        1: "filter",
        2: "group",
        }

#we have 9 filters operators (3 more for strings)
INT_OPERATOR_MAP = {
    0: operator.eq,
    1: operator.gt,
    2: operator.ge,
    3: operator.lt,
    4: operator.le,
    5: operator.ne,
    6: None,  #string contains
    7: None, #string startswith
    8: None, #string endswith
}


INT_OPERATOR_MAP_REACT = {
    8: operator.eq,
    32: operator.gt,
    64: operator.ge,
    128: operator.lt,
    256: operator.le,
    512: operator.ne,
}

AGG_MAP = {
    0: np.sum,
    1: len ,
    2: hack_min,#lambda x:np.nanmin(x.dropna()),
    3: hack_max,#lambda x:np.nanmax(x.dropna()),
    4: np.mean
}


AGG_MAP_react = {
    'sum': np.sum,
    'count': len ,
    'min': hack_min,#lambda x:np.nanmin(x.dropna()),
    'max': hack_max,#lambda x:np.nanmax(x.dropna()),
    'avg': np.mean
}

ACTION_TYPES = {
    0: 'back',
    1:'filter',
    2: 'group'
}

FILTER_LIST = ['82.108.87.7',
 '82.108',
 '75',
 '82.108.163.88',
 'login',
 'javascript',
 '00:0c:29:54:bf:79',
 'PSH',
 'reply id=0x0200',
 'BROWSER',
 'applicatio',
 '00:26:b9:2b:0b:59',
 'Echo (ping) reply',
 'NBNS',
 '42.253',
 '28',
 '82.',
 '135',
 'ping',
 'SMB',
 '82.108.84.10',
 '10.0.4.15',
 '[SYN, ACK]',
 '443',
 '192',
 '10.0.3.15',
 'Auth',
 'BOOTP',
 'auth',
 '8884',
 'SOCKS',
 '64.236.114.1',
 '82.108.87.6',
 '98.114.205.102',
 '82.108.24.168',
 '139',
 'SYN',
 '.56.52',
 '82.108.203.177',
 'HTTP',
 'RST, ACK',
 'exe',
 'Nzmx',
 '80',
 '82.108.10.107',
 'ICMP',
 '192.169.1.122',
 '17',
 '1370168419',
 'request',
 'fg',
 '192.168.1.122',
 '200',
 'Error:',
 'Not Found',
 'DATA-TEXT-LINES',
 '10.42',
 '1957',
 'DHCP',
 '82.108.0.41',
 'GET',
 '255.255.255.255',
 '94.114.205.102',
 '82.108.19.97',
 'load.php?e=1',
 '192.168.1.1',
 'icmp',
 'favicon',
 'SYN,',
 '10.0.2.15',
 '82.108.163.80',
 '121',
 'reply id',
 '3',
 '192.168.56.50',
 'Echo (ping) request',
 '43',
 'ARP',
 'tcp',
 'MEDIA',
 '1331',
 '2',
 '445',
 'honey',
 '82.108.171.155',
 '1',
 'http',
 '82.108.69',
 '1370168368',
 'ACK',
 'Who has',
 'TCP',
 '0',
 'POST',
 'destination',
 '[SYN, ACK]',
 '10.42.42.253',
 '2152',
 '10.0.2.2',
 '10.0.',
 '3feb5a6b2f',
 '82.108.56.65',
 '10.42.42.50',
 '5',
 '62',
 'Ja',
 'MZ',
 '82.108.10.174',
 '192.168.56.52',
 'RST']

KEYS=[ 'eth_dst', 'eth_src', 'highest_layer', 'info_line',
       'ip_dst', 'ip_src', 'length', 'number',
        'sniff_timestamp', 'tcp_dstport', 'tcp_srcport',
       'tcp_stream']
ACTION_TYPES_NO = len(OPERATOR_TYPE_LOOKUP)
COLS_NO = len(KEYS)
FILTER_OPS = len(INT_OPERATOR_MAP)
FILTER_TERMS_NO = len(FILTER_LIST)
AGG_FUNCS_NO=len(AGG_MAP)

empty_state_dict={"filtering":[],"grouping":[],"aggregations":[]}


def get_keys():
    return KEYS

def get_filtered_df(dataset_df, filter_list):
#legacy:
    if not filter_list:
        return dataset_df

    df = dataset_df.copy()

    for filt in filter_list:
        field = filt["field"]
        op_num = filt["condition"]
        value = filt["term"]
        
        opr = INT_OPERATOR_MAP.get(op_num)
        if op_num is not None:
            try:
                value= float(value) if df[field].dtype!='O' else value
                df = df[opr(df[field], value)]
            except:
                return df.truncate(after=-1)
        else:
            """
            if op_num==16:
                df = df[df[field].str.contains(value,na=False)]
            if op_num==2:
                df = df[df[field].str.startswith(value,na=False)]
            if op_num==4:
                df = df[df[field].str.endswith(value,na=False)]
            """
            if op_num==6:
                df = df[df[field].str.contains(value,na=False)]
            if op_num==7:
                df = df[df[field].str.startswith(value,na=False)]
            if op_num==8:
                df = df[df[field].str.endswith(value,na=False)]

    return df

def get_groupby_df(df, groupings, aggregations):

        if not groupings:
            return None,None
        
        df_gb= df.groupby(groupings)
        
        #agg_dict={'number':len} #all group-by gets the count by default in REACT-UI
        #if aggregations: #Custom aggregations: sum,count,avg,min,max
        agg_dict={}
        for agg in aggregations:
            agg_dict[agg['field']] = AGG_MAP.get(agg['type'])

        try:    
            agg_df = df_gb.agg(agg_dict)
        except:
            return None,None
        return df_gb, agg_df

def get_data_column_measures(column):
    #for each column, compute its: (1) normalized value entropy (2)Null count (3)Unique values count
    B=20
    size=len(column)
    if size ==0:
        return {"unique":0.0,"nulls":1.0,"entropy":0.0}
    n = column.isnull().sum()
    u = column.nunique()
    u_n = u/(size-n) if u !=0 else 0
    column_na=column.dropna()

    if column.dtype=='O':
        h=scipy.stats.entropy(column_na.value_counts().values)
        cna_size = len(column.dropna())
        h = h/np.log(cna_size) if cna_size > 1 else 0.0
    else:
        h= scipy.stats.entropy(np.histogram(column_na,bins=B)[0])/np.log(B)
    return {"unique":u_n,"nulls":n/size,"entropy":h}

def calc_data_layer(df):
    if len(df)==0:
        ret_dict={}
        for k in KEYS:
            ret_dict[k]={"unique":0.0,"nulls":1.0,"entropy":0.0}
        return ret_dict
        
    return df[KEYS].apply(get_data_column_measures).to_dict()

def get_grouping_measures(group_obj,agg_df):
    if group_obj is None or agg_df is None:
        return None 
    B=20
    groups_num=len(group_obj)
    if groups_num==0:
        size_var=0.0
        size_mean=0.0
    else:
        sizes=group_obj.size()
        sizes_sum=sizes.sum()
        nsizes=sizes/sizes_sum
        size_var=nsizes.var(ddof=0)
        size_mean = nsizes.mean()
        if sizes_sum>0:
            ngroups=groups_num/sizes_sum
        else:
            ngroups=0

    group_keys=group_obj.keys
    agg_keys=list(agg_df.keys())
    agg_nve_dict={}
    if agg_keys is not None:
        for ak in agg_keys:
            column = agg_df[ak]
            column_na=column.dropna()
            if column.dtype=='O':
                h=scipy.stats.entropy(column_na.value_counts().values)
                cna_size = len(column_na)
                agg_nve_dict[ak] = h/np.log(cna_size) if cna_size > 1 else 0.0
            else:
                agg_nve_dict[ak]=scipy.stats.entropy(np.histogram(column_na,bins=B)[0])/np.log(B)
    return {"group_attrs":group_keys,"agg_attrs":agg_nve_dict,"ngroups":ngroups,"size_var":size_var,"size_mean":size_mean}
    
def calc_gran_layer(group_obj,agg_df):
    #print(disp_row.display_id)
    return get_grouping_measures(group_obj,agg_df)
    
def get_raw_data(dataset_df,state_dict):
    
    filtered_df = get_filtered_df(dataset_df, state_dict["filtering"])
    gdf, agg_df= get_groupby_df(filtered_df, state_dict["grouping"], state_dict["aggregations"])
    return (filtered_df,gdf,agg_df)

def calc_display_vector(dataset_df,state_dict, ret_display=False):
    fdf,gdf,adf=get_raw_data(dataset_df,state_dict)
    data_layer=calc_data_layer(fdf)
    gran_layer=calc_gran_layer(gdf,adf)
    
    vlist=[]
    for d in data_layer.values():
        vlist+=list(d.values())
    
    if not gran_layer:
        vlist+=[-1 for k in KEYS]+[0,0,0]
    else:
        for k in KEYS:
            if k in gran_layer['agg_attrs'].keys():
                vlist.append(gran_layer['agg_attrs'][k])
            elif k in gran_layer['group_attrs']:
                vlist.append(2)
            else:
                vlist.append(-1)

        vlist+=[gran_layer['ngroups'],gran_layer['size_mean'],gran_layer['size_var']]
    
    ddict={"data_layer":data_layer,"granularity_layer":gran_layer}
    if ret_display:    
        return np.array(vlist), ddict, (fdf, adf) 
    else:
        return np.array(vlist), ddict
        
        
def random_action(atype="random"):
    
    if atype=="group":
        rtype=2
    elif atype=="filter":
        rtype=1
    elif atype=="back":
        rtype=0
    else:
        rtype=np.random.randint(ACTION_TYPES_NO)
    col=np.random.randint(COLS_NO)
    filter_op=np.random.randint(FILTER_OPS)
    filter_term=np.random.randint(FILTER_TERMS_NO)
    agg_col=np.random.randint(COLS_NO)
    agg_func=np.random.randint(AGG_FUNCS_NO)
    return [rtype,col,filter_op,filter_term,agg_col,agg_func]


