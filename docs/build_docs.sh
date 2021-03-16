cd ..
pdoc --html dvu --output-dir docs/docs
cp -r docs/docs/dvu/* docs/docs
rm -rf docs/docs/dvu
cd docs/docs
rm index.html