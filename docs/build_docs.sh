cd ..
pdoc --html data-viz-utils --output-dir docs/docs
cp -r docs/docs/data-viz-utils/* docs/docs
rm -rf docs/docs/data-viz-utils
cd docs/docs
rm index.html