#include <algorithm>
#include <codecvt>
#include <iostream>
#include <string>
#include <vector>

#include "common/wave_data.hpp"
#include "common/feature.hpp"

#include "common/utils/csrc/file_system.h"

std::wstring utf8string2wstring(std::string& str)
{
    static std::wstring_convert<std::codecvt_utf8<wchar_t> > strCnv;
    return strCnv.from_bytes(str);
}

std::string wstring2utf8string(const std::wstring& str)
{
    static std::wstring_convert<std::codecvt_utf8<wchar_t> > strCnv;
    return strCnv.to_bytes(str);
}

std::string wchar2utf8string(wchar_t &str)
{
    static std::wstring_convert<std::codecvt_utf8<wchar_t> > strCnv;
    return strCnv.to_bytes(str);
}

std::vector<std::string> StringSplit(const std::string &s, const std::string &seperator)
{	
    std::vector<std::string> result;
    typedef std::string::size_type string_size;

    int swidth = seperator.size();
    string_size i = 0, j = 0;

    while ( (j + swidth) <= s.size() )
    {	
        if ( s.substr(j, swidth) == seperator )
        {   
            if ( j-i > 0 )
            {
                result.push_back( s.substr(i,(j-i)) );
            }
            i = j + swidth;
            j = i;
        }
        else
        {   
            j ++;
        }
    }

    if ( i < s.size() )
    {   
        result.push_back( s.substr(i, s.size()-i) );
    }
    
    return result;
}


int main(int argc, char **argv) {

    // std::string output_str = "ASR: ^他妈的_F-脏话$";
    std::string output_str = "ASR: ^抱歉，你似乎没有说话$";
    std::wstring output_str_w;
    std::cout << "\033[0;31m" << "Get Infor: " << output_str << "\033[0;39m" << std::endl;
    
    // // test
    // for( unsigned int i = 0; i < output_str.length(); i++ )
    // {
    //     printf("%02x ", output_str[i]);
    //     std::cout << "\033[0;31m" << "i: " << i << ", char: " << output_str[i] << "\033[0;39m" << std::endl;
        
    // }
    
    // // test
    // output_str_w = utf8string2wstring(output_str);
    // for( unsigned int i = 0; i < output_str_w.length(); i++ )
    // {
    //     std::cout << "\033[0;31m" << "i: " << i << ", char: " << wchar2utf8string(output_str_w[i]) << "\033[0;39m" << std::endl;
        
    // }

    std::vector<std::string> output_str_vec = StringSplit(output_str, " ");
    for( unsigned int i = 0; i < output_str_vec.size(); i++ )
    {   
        output_str_w = utf8string2wstring(output_str_vec[i]);

        int show_times = output_str_w.length() / 5 + 1;
        std::cout << "\033[0;31m" << "length: " << output_str_w.length() << ", show_times: " << show_times << "\033[0;39m" << std::endl;

        for ( int j = 0; j < show_times; j++ )
        {
            // 叠加数据打印
            if ( j * 5 >= output_str_w.length())
            {
                break;
            }
            
            std::cout << "\033[0;31m" << "Get SubInfor: " << wstring2utf8string(output_str_w.substr(j*5, 5)) << "\033[0;39m" << std::endl;
        }
    }

    return 0;
}