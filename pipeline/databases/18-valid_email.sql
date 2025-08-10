-- Write a SQL script that creates a trigger that resets the
-- attribute valid_email only when the email has been changed.
-- Context: Nothing related to MySQL, but perfect for user email
-- validation - distribute the logic to the database itself!

delimiter //
CREATE TRIGGER update_email
	BEFORE UPDATE
	ON users
	FOR EACH ROW
	BEGIN
  		IF STRCMP(old.email, new.email) <> 0 THEN
			SET new.valid_email = 0;
		END IF;
	END //
delimiter ;
