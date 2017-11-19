files=('train-images-idx3-ubyte.gz' 'train-labels-idx1-ubyte.gz' 't10k-images-idx3-ubyte.gz' 't10k-labels-idx1-ubyte.gz')
for file in ${files[@]}; do
    wget "http://yann.lecun.com/exdb/mnist/$file"
    gunzip "$file"
done

