def get_security_type(security):
    exchange = security[-4:]
    code = security[:-5]
    if code.isdigit():
        if exchange == "XSHG":
            if code >= "600000" or code[0] == "9":
                return "stock"
            elif code >= "500000":
                return "fund"
            elif code[0] == "0":
                return "index"
            elif len(code) == 8 and code[0] == '1':
                return "option"
        elif exchange == "XSHE":
            if code[0] == "0" or code[0] == "2" or code[:3] == "300":
                return "stock"
            elif code[:3] == "399":
                return "index"
            elif code[0] == "1":
                return "fund"
        else:
            raise Exception("找不到标的%s" % security)
    else:
        if exchange in ("XSGE", "XDCE", "XZCE", "XINE", "CCFX"):
            if len(code) > 6:
                return "option"
            return "future"
    return 0
