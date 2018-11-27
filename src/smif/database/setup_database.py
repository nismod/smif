# sets up the nismod database
# uses the SFTP server to get datafiles for database

import re, subprocess, os
from os import listdir as listdir
from os.path import dirname, realpath, join, isfile
import psycopg2
from psycopg2.extras import DictCursor


def run_migrations(migrations_dir, migration_direction):
	"""Run specified database migrations
	"""

	# run migrations given parameter passed
	if migration_direction == 'up':
		ran = up_migrations(migrations_dir)
	elif migration_direction == 'down':
		ran = down_migrations(migrations_dir)

	return ran


def down_migrations(working_path):
	"""Run down migrations to resolve database to blank
	"""

	# run key file to remove foreign keys from database
	file = 'remove-foreign_keys.sql'
	# check file exists
	if isfile(join(working_path, file)):
		# run sql file silently
		subprocess.run(['psql', '-U', 'vagrant', '-d', 'nismod_smif_conf', '-q', '-f', join(working_path, file)])
	else:
		return False

	# loop through files to remove tables
	for file in listdir(working_path):

		# check if file is a down migration
		if file[0:4] == 'down':
			# run sql file silently
			subprocess.run(['psql', '-U', 'vagrant', '-d', 'nismod_smif_conf', '-q', '-f', join(working_path, file)])

	return True


def up_migrations(working_path):
	"""Run up migrations to build database
	"""

	# declare name of foreign key file
	# run key file to add foreign keys between relations
	fky_file = 'add-foreign_keys.sql'

	# check file exists - if not stop now
	if isfile(join(working_path, fky_file)) is False:
		return False

	# loop through files
	for file in listdir(working_path):

		# check if file is a down migration
		if file[0:2] == 'up':
			# run sql file silently
			subprocess.run(['psql', '-U', 'vagrant', '-d', 'nismod_smif_config', '-q', '-f', join(working_path, file)])

	# run key file to add foreign keys between relations
	# run sql file silently
	subprocess.run(['psql', '-U', 'vagrant', '-d', 'nismod_smif_conf', '-q', '-f', join(working_path, fky_file)])

	return True


def main():
	"""Run database setup process
	"""

	# get the base directory and build the path for the database config files
	config_dir = dirname(realpath(__file__))

	# run database provision file - adds extensions to database if required
	#subprocess.run(['sudo', 'sh', config_dir + '/provision-db.sh'])

	# build the database if it does not exist
	subprocess.run(['psql', '-U', 'vagrant', '-d', 'nismod_smif_conf', '-q', '-c', 'CREATE DATABASE IF DOES NOT EXIST nismod_smif_config'])

	# run migrations - delete existing relations than add new set
	# run the migrations - down first in case anything exists already
	# run migrations - down
	run_migrations(config_dir + '/migrations', 'down')

	# run the migrations - up to build the database
	# run migrations - up
	run_migrations(config_dir + '/migrations', 'up')

	# confirm migrations have been run
	subprocess.run(['sudo', 'echo', 'Finished building database'])

# run main when file called
main()