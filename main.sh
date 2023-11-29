dir='common'
file_list=`cat $dir/list.txt`
write_file_name="${dir}/name.py"
tag=''
prod_check='test'

while getopts ts-: opt; do
    optarg="${!OPTIND}"
    [[ "$opt" = - ]] && opt="-${OPTARG}"
    
    case ${opt} in
        s|-state)
            prod_check="${optarg}"
            shift
            ;;
        t|-tag)
            tag="${optarg}"
            shift
            ;;
    esac
done

if [ -z $tag ]; then
    echo '1: name update'
    echo '2: data update and learn'
    echo '3: learn'
    echo '4: optuna learn'
    echo '5: simulation'

    read -p "Enter 1,2,3,4,5 > " tag
fi

if [ !$tag = '1' ] && [ !$tag = '2' ] && [ !$tag = '3' ] && [ !$tag = '4' ] && [ !$tag = '5' ]; then
    echo "Wrong number"
    exit 1
fi

rm -rf $write_file_name
echo 'class Name:' >> $write_file_name
echo '    def __init__( self ):' >> $write_file_name

for file_name in $file_list; do
    base='        self.'
    ARR=(${file_name//./ })
    name=${ARR[0]}
    echo "$base$name = \"$name\"" >> $write_file_name
    done

cp -r $dir data_analyze/
cp -r $dir learn/

if [ $tag = "2" ]; then
    mpiexec -n 3 python main.py -u True
fi

if [ $tag = "3" ]; then
    python main.py -l True -s "${prod_check}"
fi

if [ $tag = "4" ]; then
    python main.py -o True -s "${prod_check}"
fi

if [ $tag = "5" ]; then
    python main.py
fi

rm -rf storage
