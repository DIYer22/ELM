# -*- coding: utf-8 -*-


from __future__ import unicode_literals

import time

def performance(f):
    '''
    性能测试装饰器
    '''
    def fn(*args, **kw):
        t1 = time.time()
        r = f(*args, **kw)
        t2 = time.time()
        print 'call %s() in %fs' % (f.__name__, (t2 - t1))
        return r
    return fn


def fLog(display=0, turn_on=1):
    '''
    DEBUG用的装饰器 display:显示参数, turn_on:是否使用装饰器
    '''
    def _log(f):
        def fn(*args, **kw):
            r = f(*args, **kw)
            if not turn_on:
                return r
            if display:
                args_txt = str(args)
                if len(args) == 1:
                    args_txt = args_txt[:-2] + ')'
                print '\t'*5+'%s%s return %s' % (f.__name__, args_txt, str(r))
            else:
                print '\t'*5+'%s() is called' % (f.__name__)
                
            return r
        return fn
    return _log

def log(d):
    '''打印
    '''
    show_len = 20
    if isinstance(d,list):
        for i in d:
            strr = str(i)
            if len(strr) > show_len:
                print strr[:show_len] + '...'
            else:
                print strr
        return
            
    if isinstance(d,dict):
        for i in d:
            strr = str(i)+' = '+str(d[i])
            if len(strr) > show_len:
                print strr[:show_len] + '...'
            else:
                print strr          
        return
    print d

#url = 'http://www.baidu.com/s?wd=ip'
#
#creat_file(download(url),'1.html')





