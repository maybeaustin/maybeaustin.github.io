
###
# Developing
#./hugo server -D
#./hugo
###

###
# Publish
###
rm -r public
./hugo
cp -a .git public/.git
cd public
git branch -D master
git branch master
git checkout master
git add -A
git commit -m"update website"
git remote add origin https://github.com/austindavidbrown/austindavidbrown.github.io.git
git push origin master --force